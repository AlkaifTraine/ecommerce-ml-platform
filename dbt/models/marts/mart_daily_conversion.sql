-- Daily conversion with a rolling baseline - the drift monitor's input table.
--
-- The interesting column is `conv_vs_baseline_pct`. Feature distributions in
-- this dataset barely move (price PSI 0.0024, category mix PSI 0.0116), so a
-- feature-drift dashboard stays green while conversion falls ~26% relative
-- between October and November. Tracking the LABEL rate against its own
-- trailing baseline is what actually catches that.
--
-- The trailing window deliberately excludes the current day, so a bad day
-- cannot flatter its own baseline.

with daily as (

    select
        d.date_key,
        d.day_of_week,
        d.is_weekend,
        d.is_quarantined,
        a.sessions,
        a.purchases,
        a.carts,
        a.events,
        a.revenue,
        a.avg_price,
        a.buy_per_cart,
        100.0 * a.purchases / nullif(a.sessions, 0) as conv_pct

    from {{ source('warehouse', 'agg_daily') }} a
    join {{ source('warehouse', 'dim_date') }}  d using (date_key)

),

with_baseline as (

    select
        *,
        -- 7 prior days, current day excluded
        avg(case when is_quarantined then null else conv_pct end) over (
            order by date_key rows between 7 preceding and 1 preceding
        ) as conv_baseline_pct,

        avg(case when is_quarantined then null else events end) over (
            order by date_key rows between 7 preceding and 1 preceding
        ) as events_baseline

    from daily

)

select
    date_key,
    day_of_week,
    is_weekend,
    is_quarantined,
    sessions,
    purchases,
    carts,
    events,
    revenue,
    avg_price,
    buy_per_cart,
    conv_pct,
    conv_baseline_pct,

    case
        when conv_baseline_pct is null then null
        else 100.0 * (conv_pct - conv_baseline_pct) / nullif(conv_baseline_pct, 0)
    end as conv_vs_baseline_pct,

    case
        when events_baseline is null then null
        else events / nullif(events_baseline, 0)
    end as volume_vs_baseline,

    -- Volume is the reliable outage signal. buy/cart is not usable as a global
    -- threshold: it is ~69% in October and ~34% in November because cart
    -- events are under-recorded in October, so there is no stable baseline.
    case
        when purchases = 0                                    then 'NO_PURCHASES'
        when events / nullif(events_baseline, 0) > 2.0        then 'VOLUME_ANOMALY'
        when conv_baseline_pct is not null
             and abs(conv_pct - conv_baseline_pct)
                 / nullif(conv_baseline_pct, 0) > 0.25        then 'CONVERSION_SHIFT'
        else 'OK'
    end as data_quality_flag

from with_baseline
order by date_key
