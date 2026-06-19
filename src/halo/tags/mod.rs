//! Runtime tag cache — engine `global_tag_instances[]` analog. Holds
//! parsed tag data Arc'd by FOURCC + path so any code that needs a
//! tag asks the cache once, gets a shared `Arc<LoadedTag>` back, and
//! never re-parses.

pub mod cache;

pub use cache::{tag_get, tag_init};
