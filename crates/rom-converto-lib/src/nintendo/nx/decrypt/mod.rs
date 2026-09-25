//! Encrypted NSP/XCI -> NxEmu DNSP/DXCI.
//!
//! NxEmu ships no decryption code and loads `.dnsp` / `.dxci` instead:
//! the same containers with every NCA rewritten as plaintext (see
//! [`nca`]) and the gamecard magic changed from `HEAD` to `DXCI`.
//! Decryption preserves every size, so the output mirrors the input
//! byte for byte except for the rewritten NCAs, the HFS0 entry hashes
//! that cover their headers, and the gamecard header's root-HFS0 hash.
//! Tickets, certificates, and CNMT XMLs are copied through untouched.

pub mod bucket_tree;
pub mod nca;

use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::nintendo::nx::constants::HFS0_ENTRY_SIZE;
use crate::nintendo::nx::container::{
    ContainerKind, DXCI_MAGIC, list_container, read_xci_hfs0_offset,
};
use crate::nintendo::nx::error::{NxError, NxResult};
use crate::nintendo::nx::keys::KeySet;
use crate::nintendo::nx::meta::merge_inline_tickets;
use crate::nintendo::nx::models::hfs0::Hfs0;
use crate::nintendo::nx::util::copy_range;
use crate::util::bytes::u64_le;
use crate::util::pread::file_read_exact_at;
use crate::util::{AtomicProgress, CancelToken, ProgressReporter, run_scratch_write};
use nca::NcaPlainPlan;

const XCI_MAGIC_OFFSET: usize = 0x100;
const XCI_ROOT_HEADER_SIZE_OFFSET: usize = 0x138;
const XCI_ROOT_HEADER_HASH_OFFSET: usize = 0x140;
/// Gamecard header length; the root HFS0 can never start inside it.
const XCI_HEADER_SIZE: u64 = 0x200;
const HFS0_ENTRY_HASH_OFFSET: usize = 0x20;

/// Decrypts the NSP or XCI at `input` into the DNSP/DXCI at `output`.
/// Tickets bundled in the container supply the title keys of
/// rights-protected NCAs; `keys` must hold the header key and the
/// key-area/titlekek keys for each NCA's key generation.
///
/// # Errors
/// Fails if `input` is compressed (NSZ/XCZ), if a needed key is
/// missing, if an NCA already carries the `DNCA` magic, if a section
/// uses XTS (encrypted input only) or a sparse layer, or on the
/// underlying I/O and parsing errors. hactool plaintext NCAs are
/// accepted and only get their headers rewritten.
pub fn decrypt_container(
    input: &Path,
    output: &Path,
    keys: &KeySet,
    progress: &dyn ProgressReporter,
    cancel: Option<&CancelToken>,
) -> NxResult<()> {
    let listing = list_container(input)?;
    if listing.kind.is_compressed() {
        return Err(NxError::CompressedInputUnsupported(input.to_path_buf()));
    }
    let mut keys = keys.clone();
    merge_inline_tickets(input, &listing, &mut keys);

    let file = Arc::new(File::open(input)?);
    let total = file.metadata()?.len();
    let segments = match listing.kind {
        ContainerKind::Nsp => nsp_segments(&file, &listing.entries, &keys)?,
        ContainerKind::Xci => xci_segments(&file, input, &keys, progress)?,
        ContainerKind::Nsz | ContainerKind::Xcz => unreachable!("compressed kinds rejected above"),
    };

    let never = CancelToken::new();
    let cancel = cancel.unwrap_or(&never);
    let mut out = BufWriter::new(File::create(output)?);
    write_segments(&file, total, &segments, &mut out, progress, cancel)?;
    out.flush()?;
    Ok(())
}

/// Async twin of [`decrypt_container`]: writes to a scratch sibling of
/// `output` and publishes it on success; a cancelled or failed run
/// leaves nothing behind.
pub async fn decrypt_container_async(
    input: PathBuf,
    output: PathBuf,
    keys: KeySet,
    progress: &dyn ProgressReporter,
    cancel: CancelToken,
) -> NxResult<()> {
    let total = tokio::fs::metadata(&input).await?.len();
    progress.start(total, "Decrypting Switch container");
    run_scratch_write(
        &output,
        true,
        progress,
        &cancel,
        move |write_path, bytes_done, cancel| {
            let proxy = AtomicProgress {
                counter: bytes_done,
            };
            decrypt_container(&input, &write_path, &keys, &proxy, Some(&cancel))
        },
    )
    .await
}

/// One byte range of the output that differs from the input: either
/// replacement bytes (patched headers) or an NCA to rewrite.
struct Segment {
    abs: u64,
    len: u64,
    kind: SegmentKind,
}

enum SegmentKind {
    Bytes(Vec<u8>),
    Nca(NcaPlainPlan),
}

fn is_nca(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".nca")
}

fn nsp_segments(
    file: &Arc<File>,
    entries: &[crate::nintendo::nx::container::ContainerEntry],
    keys: &KeySet,
) -> NxResult<Vec<Segment>> {
    let mut segments = Vec::new();
    for entry in entries.iter().filter(|e| is_nca(&e.name)) {
        let plan = NcaPlainPlan::open(
            file.clone(),
            entry.abs_offset,
            entry.size,
            &entry.name,
            keys,
        )?;
        segments.push(Segment {
            abs: entry.abs_offset,
            len: entry.size,
            kind: SegmentKind::Nca(plan),
        });
    }
    Ok(segments)
}

/// Plans an XCI rewrite: the gamecard prefix with the `DXCI` magic and
/// refreshed root-HFS0 hash, the root and partition headers with
/// refreshed entry hashes, and every NCA in every partition. NxEmu
/// only reads the `update` partition to install firmware and skips
/// any update NCA it cannot open, while its system NCAs may need
/// newer keys or layouts than the game itself, so an update NCA that
/// cannot be planned is copied through with a warning instead of
/// failing the conversion. I/O errors stay fatal.
fn xci_segments(
    file: &Arc<File>,
    input: &Path,
    keys: &KeySet,
    progress: &dyn ProgressReporter,
) -> NxResult<Vec<Segment>> {
    let hfs0_off = {
        let mut probe = File::open(input)?;
        read_xci_hfs0_offset(&mut probe)?
    };
    if hfs0_off < XCI_HEADER_SIZE {
        return Err(NxError::InvalidXci);
    }
    let mut reader = BufReader::new(File::open(input)?);
    reader.seek(SeekFrom::Start(hfs0_off))?;
    let root = Hfs0::read(&mut reader)?;
    let mut root_header = read_at(file, hfs0_off, root.data_section_offset - hfs0_off)?;

    let mut segments = Vec::new();
    for (index, partition) in root.files.iter().enumerate() {
        let part_abs = root.data_section_offset + partition.data_offset;
        reader.seek(SeekFrom::Start(part_abs))?;
        let sub = Hfs0::read(&mut reader)?;
        let mut sub_header = read_at(file, part_abs, sub.data_section_offset - part_abs)?;

        for (i, entry) in sub.files.iter().enumerate() {
            if !is_nca(&entry.name) {
                continue;
            }
            let abs = sub.data_section_offset + entry.data_offset;
            let plan = match NcaPlainPlan::open(file.clone(), abs, entry.size, &entry.name, keys) {
                Err(err)
                    if partition.name.eq_ignore_ascii_case("update")
                        && !matches!(err, NxError::IoError(_)) =>
                {
                    progress.warn(&format!(
                        "update partition NCA {} left encrypted: {err}",
                        entry.name
                    ));
                    continue;
                }
                plan => plan?,
            };
            let prefix = plan.plain_prefix(file, abs, u64::from(entry.hashed_region_size))?;
            set_entry_hash(&mut sub_header, i, &prefix);
            segments.push(Segment {
                abs,
                len: entry.size,
                kind: SegmentKind::Nca(plan),
            });
        }

        // The root entry hashes the partition header; a region that
        // ran into file data would need the rewritten bytes too.
        let hashed = partition.hashed_region_size as usize;
        if hashed > sub_header.len() {
            return Err(NxError::InvalidXci);
        }
        set_entry_hash(&mut root_header, index, &sub_header[..hashed]);
        segments.push(Segment {
            abs: part_abs,
            len: sub_header.len() as u64,
            kind: SegmentKind::Bytes(sub_header),
        });
    }

    let mut prefix = read_at(file, 0, hfs0_off)?;
    prefix[XCI_MAGIC_OFFSET..XCI_MAGIC_OFFSET + 4].copy_from_slice(&DXCI_MAGIC);
    let hashed = (u64_le(&prefix, XCI_ROOT_HEADER_SIZE_OFFSET) as usize).min(root_header.len());
    let root_hash = Sha256::digest(&root_header[..hashed]);
    prefix[XCI_ROOT_HEADER_HASH_OFFSET..XCI_ROOT_HEADER_HASH_OFFSET + 32]
        .copy_from_slice(&root_hash);

    segments.push(Segment {
        abs: hfs0_off,
        len: root_header.len() as u64,
        kind: SegmentKind::Bytes(root_header),
    });
    segments.push(Segment {
        abs: 0,
        len: hfs0_off,
        kind: SegmentKind::Bytes(prefix),
    });
    Ok(segments)
}

fn read_at(file: &File, abs: u64, len: u64) -> NxResult<Vec<u8>> {
    let mut buf = vec![0u8; len as usize];
    file_read_exact_at(file, &mut buf, abs)?;
    Ok(buf)
}

/// Stores the SHA-256 of `hashed` in HFS0 entry `index` of `header`.
fn set_entry_hash(header: &mut [u8], index: usize, hashed: &[u8]) {
    let at = 0x10 + index * HFS0_ENTRY_SIZE + HFS0_ENTRY_HASH_OFFSET;
    header[at..at + 32].copy_from_slice(&Sha256::digest(hashed));
}

/// Streams the input to `out`, copying every byte outside a segment
/// verbatim and emitting each segment in its place.
fn write_segments<W: Write>(
    file: &File,
    total: u64,
    segments: &[Segment],
    out: &mut W,
    progress: &dyn ProgressReporter,
    cancel: &CancelToken,
) -> NxResult<()> {
    let mut order: Vec<&Segment> = segments.iter().collect();
    order.sort_by_key(|s| s.abs);

    let mut pos = 0u64;
    for segment in order {
        if segment.abs < pos || segment.abs + segment.len > total {
            return Err(NxError::OverlappingEntries);
        }
        copy_range(file, pos, segment.abs - pos, out, progress, cancel)?;
        match &segment.kind {
            SegmentKind::Bytes(bytes) => {
                out.write_all(bytes)?;
                progress.inc(bytes.len() as u64);
            }
            SegmentKind::Nca(plan) => {
                plan.write_plain(file, segment.abs, out, progress, Some(cancel))?
            }
        }
        pos = segment.abs + segment.len;
    }
    copy_range(file, pos, total - pos, out, progress, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nintendo::nx::constants::NCA_HEADER_SIZE;
    use crate::nintendo::nx::crypto::aes_xts::decrypt_nca_header;
    use crate::nintendo::nx::models::cnmt::CNMT_CONTENT_TYPE_PROGRAM;
    use crate::nintendo::nx::models::hfs0::hash_first_chunk;
    use crate::nintendo::nx::test_fixtures::{
        TEST_HEADER_KEY, build_meta_nca, build_test_nsp, build_test_xci, synthetic_keyset,
    };
    use crate::util::NoProgress;
    use tempfile::TempDir;

    fn write_temp(dir: &TempDir, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join(name);
        File::create(&path).unwrap().write_all(bytes).unwrap();
        path
    }

    fn expected_plain_meta_nca(dir: &TempDir, encrypted: &[u8]) -> Vec<u8> {
        let file = Arc::new(File::open(write_temp(dir, "x.nca", encrypted)).unwrap());
        let plan = NcaPlainPlan::open(
            file.clone(),
            0,
            encrypted.len() as u64,
            "x.nca",
            &synthetic_keyset(),
        )
        .unwrap();
        let mut out = Vec::new();
        plan.write_plain(&file, 0, &mut out, &NoProgress, None)
            .unwrap();
        out
    }

    #[test]
    fn nsp_rewrites_ncas_in_place_and_keeps_other_files() {
        let nca = build_meta_nca(
            0x0100_0000_0000_1000,
            1,
            CNMT_CONTENT_TYPE_PROGRAM,
            &[[0xAB; 16]],
        );
        let tik = vec![0x5Au8; 0x2C0];
        let nsp = build_test_nsp(&[
            (
                "0000000000000000000000000000000a.cnmt.nca".into(),
                nca.clone(),
            ),
            ("title.tik".into(), tik.clone()),
        ]);
        let dir = TempDir::new().unwrap();
        let input = write_temp(&dir, "game.nsp", &nsp);
        let output = dir.path().join("game.dnsp");

        decrypt_container(&input, &output, &synthetic_keyset(), &NoProgress, None).unwrap();

        let out = std::fs::read(&output).unwrap();
        assert_eq!(out.len(), nsp.len());
        let nca_start = nsp.len() - tik.len() - nca.len();
        assert_eq!(&out[..nca_start], &nsp[..nca_start]);
        assert_eq!(
            &out[nca_start..nca_start + nca.len()],
            expected_plain_meta_nca(&dir, &nca)
        );
        assert_eq!(&out[nca_start + nca.len()..], tik.as_slice());

        let mut header = [0u8; NCA_HEADER_SIZE];
        header.copy_from_slice(&nca[..NCA_HEADER_SIZE]);
        decrypt_nca_header(&mut header, &TEST_HEADER_KEY).unwrap();
        let plain = &out[nca_start..nca_start + NCA_HEADER_SIZE];
        assert_eq!(&plain[0x200..0x204], b"DNCA");
        assert_eq!(&plain[0x204..0x280], &header[0x204..0x280]);
    }

    #[test]
    fn xci_patches_magic_and_refreshes_hfs0_hashes() {
        let nca = build_meta_nca(
            0x0100_0000_0000_2000,
            3,
            CNMT_CONTENT_TYPE_PROGRAM,
            &[[0xCD; 16]],
        );
        let xci = build_test_xci(&[(
            "0000000000000000000000000000000b.cnmt.nca".into(),
            nca.clone(),
        )]);
        let dir = TempDir::new().unwrap();
        let input = write_temp(&dir, "game.xci", &xci);
        let output = dir.path().join("game.dxci");

        decrypt_container(&input, &output, &synthetic_keyset(), &NoProgress, None).unwrap();

        let out = std::fs::read(&output).unwrap();
        assert_eq!(out.len(), xci.len());
        assert_eq!(&out[0x100..0x104], b"DXCI");

        let hfs0_off = u64_le(&out, 0x130) as usize;
        let root_size = u64_le(&out, 0x138) as usize;
        let root_hash: [u8; 32] = Sha256::digest(&out[hfs0_off..hfs0_off + root_size]).into();
        assert_eq!(&out[0x140..0x160], &root_hash);

        let mut reader = std::io::Cursor::new(&out);
        reader.seek(SeekFrom::Start(hfs0_off as u64)).unwrap();
        let root = Hfs0::read(&mut reader).unwrap();
        let secure = root.files.iter().find(|f| f.name == "secure").unwrap();
        let part_abs = root.data_section_offset + secure.data_offset;
        reader.seek(SeekFrom::Start(part_abs)).unwrap();
        let sub = Hfs0::read(&mut reader).unwrap();
        let sub_header = &out[part_abs as usize..sub.data_section_offset as usize];
        assert_eq!(
            secure.sha256,
            hash_first_chunk(sub_header, secure.hashed_region_size)
        );

        let entry = &sub.files[0];
        let nca_abs = (sub.data_section_offset + entry.data_offset) as usize;
        let plain = &out[nca_abs..nca_abs + entry.size as usize];
        assert_eq!(plain, expected_plain_meta_nca(&dir, &nca));
        assert_eq!(
            entry.sha256,
            hash_first_chunk(plain, entry.hashed_region_size)
        );

        // Everything outside the patched headers and the NCA is untouched.
        assert_eq!(&out[0x160..hfs0_off], &xci[0x160..hfs0_off]);
        assert_eq!(&out[nca_abs + plain.len()..], &xci[nca_abs + plain.len()..]);
    }

    /// hactool `--plaintext` output: the XTS-decrypted header as is
    /// (`NCA3` magic, FS headers still claiming CTR) over decrypted
    /// sections. Only the header rewrite is left to do, and no key is
    /// needed for it.
    #[test]
    fn hactool_plaintext_nca_gets_only_its_header_rewritten() {
        let nca = build_meta_nca(
            0x0100_0000_0000_3000,
            1,
            CNMT_CONTENT_TYPE_PROGRAM,
            &[[0xEF; 16]],
        );
        let dir = TempDir::new().unwrap();
        let expected = expected_plain_meta_nca(&dir, &nca);
        let mut hactool = expected.clone();
        let mut header = [0u8; NCA_HEADER_SIZE];
        header.copy_from_slice(&nca[..NCA_HEADER_SIZE]);
        decrypt_nca_header(&mut header, &TEST_HEADER_KEY).unwrap();
        hactool[..NCA_HEADER_SIZE].copy_from_slice(&header);
        assert_eq!(&hactool[0x200..0x204], b"NCA3");

        let file = Arc::new(File::open(write_temp(&dir, "h.nca", &hactool)).unwrap());
        let plan = NcaPlainPlan::open(
            file.clone(),
            0,
            hactool.len() as u64,
            "h.nca",
            &KeySet::default(),
        )
        .unwrap();
        let mut out = Vec::new();
        plan.write_plain(&file, 0, &mut out, &NoProgress, None)
            .unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn compressed_input_is_rejected() {
        let dir = TempDir::new().unwrap();
        let input = write_temp(&dir, "game.nsz", &build_test_nsp(&[]));
        let err = decrypt_container(
            &input,
            &dir.path().join("game.dnsp"),
            &synthetic_keyset(),
            &NoProgress,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, NxError::CompressedInputUnsupported(_)));
    }
}
