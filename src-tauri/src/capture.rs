//! Screen capture for the vision feature.
//!
//! Grab the **primary monitor**, downscale so the long edge is `<= MAX_LONG_EDGE`,
//! JPEG-encode at `JPEG_QUALITY`, and return it as base64 (no `data:` prefix).
//! Our overlay window is `contentProtected: true`, so it stays OUT of this capture
//! even while visible — no "the model sees its own answer" loop.
//!
//! `xcap::Monitor::capture_image()` returns an RGBA buffer bundled with xcap's own
//! `image` version; we bridge it to our `image` crate through raw bytes
//! (`into_raw` → `from_raw`) so a version skew between the two can never bite.
//!
//! Blocking (xcap grabs synchronously) — call it from `spawn_blocking`.

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, ExtendedColorType, ImageBuffer, ImageEncoder, Rgba};

/// Long-edge cap. Enough to read on-screen code; ~1000 image tokens on gpt-4o-mini
/// at `detail:"high"`. Bigger buys little accuracy and costs roughly linearly more.
const MAX_LONG_EDGE: u32 = 1500;
/// JPEG quality. 75 keeps text crisp while shrinking the payload well below PNG.
const JPEG_QUALITY: u8 = 75;

/// Capture the primary monitor as a downscaled base64 JPEG (no `data:` prefix).
pub fn capture_primary_jpeg_base64() -> Result<String> {
    let monitor = xcap::Monitor::all()
        .map_err(|e| anyhow!("enumerate monitors: {e}"))?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no monitors found"))?;

    let shot = monitor
        .capture_image()
        .map_err(|e| anyhow!("capture failed: {e}"))?;
    let (w, h) = (shot.width(), shot.height());

    // Bridge xcap's RGBA buffer to our `image` crate via raw bytes.
    let rgba: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(w, h, shot.into_raw())
        .ok_or_else(|| anyhow!("captured buffer size mismatch ({w}x{h})"))?;

    let rgb = DynamicImage::ImageRgba8(rgba)
        // `resize` fits WITHIN the box and only ever shrinks — aspect preserved.
        .resize(MAX_LONG_EDGE, MAX_LONG_EDGE, FilterType::Triangle)
        .to_rgb8();

    let mut buf = Vec::new();
    JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY)
        .write_image(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            ExtendedColorType::Rgb8,
        )
        .context("jpeg encode")?;

    tracing::debug!(w, h, jpeg_bytes = buf.len(), "captured primary monitor");
    Ok(STANDARD.encode(&buf))
}
