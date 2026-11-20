use super::*;
use forward::StatisticsStorage::{Global, Registers};

#[test]
fn online_register_statistics_default_and_narrow_scope_are_explicit() -> Result<()> {
    assert!(!crate::ptx::mixed::MixedOptions::default().online_attention_register_statistics);
    for batch in [8, 16, 32] {
        assert!(register_statistics_scope(
            Shape::new(batch, 8)?,
            [leaf::ReadLayout::SequenceMajor; 4]
        ));
    }
    assert!(!register_statistics_scope(
        Shape::new(64, 8)?,
        [leaf::ReadLayout::SequenceMajor; 4]
    ));
    assert!(!register_statistics_scope(
        Shape::new(8, 8)?,
        [leaf::ReadLayout::HeadMajor; 4]
    ));
    assert!(!register_statistics_scope(
        Shape::new(16, 4)?,
        [leaf::ReadLayout::SequenceMajor; 4]
    ));
    let mut calls = Vec::new();
    let (_, mode, rejection) = select_statistics_pair(false, true, |mode| {
        calls.push(mode);
        Ok(17)
    })
    .expect("Global default");
    assert_eq!(calls, [Global]);
    assert_eq!(mode, Global);
    assert!(rejection.is_none());
    Ok(())
}
#[test]
fn online_register_statistics_whole_pair_compile_and_resource_fallback() -> Result<()> {
    for reason in [Rejection::Compilation, Rejection::Resources] {
        let mut calls = Vec::new();
        let (pair, mode, rejection) = select_statistics_pair(true, true, |mode| {
            calls.push(mode);
            if mode == Registers {
                Err(reason.clone())
            } else {
                Ok([11, 22, 33])
            }
        })
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!(calls, [Registers, Global]);
        assert_eq!(mode, Global);
        assert_eq!(pair, [11, 22, 33]);
        assert_eq!(format!("{:?}", rejection), format!("{:?}", Some(reason)));
    }
    let mut calls = Vec::new();
    let result = select_statistics_pair::<[usize; 3]>(true, true, |mode| {
        calls.push(mode);
        Err(Rejection::Compilation)
    });
    assert!(matches!(result, Err(Rejection::Compilation)));
    assert_eq!(calls, [Registers, Global]);
    Ok(())
}
#[test]
fn online_register_statistics_success_and_unmatched_scope_preserve_pair_mode() -> Result<()> {
    let mut calls = Vec::new();
    let (pair, mode, rejected) = select_statistics_pair(true, true, |mode| {
        calls.push(mode);
        Ok([1, 2, 3])
    })
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    assert_eq!(calls, [Registers]);
    assert_eq!(mode, Registers);
    assert_eq!(pair, [1, 2, 3]);
    assert!(rejected.is_none());
    calls.clear();
    let (_, mode, rejected) = select_statistics_pair(true, false, |mode| {
        calls.push(mode);
        Ok([1, 2, 3])
    })
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    assert_eq!(calls, [Global]);
    assert_eq!(mode, Global);
    assert!(matches!(rejected, Some(Rejection::UnsupportedLayout)));
    Ok(())
}
