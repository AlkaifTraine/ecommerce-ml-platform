-- The quarantined days MUST be flagged by the automated detector.
--
-- This is a regression test on the detection logic, not on the data. The
-- quarantine list in config.py was derived by review; if a future change to
-- mart_daily_conversion's thresholds stops flagging 2019-11-15..17, the
-- monitor has silently lost the ability to catch the very incident it was
-- built for, and this test fails.
--
-- 2019-11-14 is deliberately NOT required here: its volume ratio (~2.0x) sits
-- exactly on the threshold, so it is quarantined by human review rather than
-- by the detector. Asserting the detector catches it would be asserting a
-- coincidence.

select
    date_key,
    events,
    purchases,
    volume_vs_baseline,
    data_quality_flag

from {{ ref('mart_daily_conversion') }}

where date_key in (date '2019-11-15', date '2019-11-16', date '2019-11-17')
  and data_quality_flag = 'OK'
