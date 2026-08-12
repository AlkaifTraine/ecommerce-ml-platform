-- RFM-style segmentation over full user history.
--
-- This is the mart that justifies having a warehouse at all: it needs every
-- session a user ever had, and the OLTP store keeps only seven days. The
-- warehouse measured that 256,958 repeat buyers - 37% of all buyers - produce
-- 73.8% of revenue, which is the number that makes "who do we spend the
-- retargeting budget on" a concrete question rather than a slogan.

with bounds as (
    select max(session_end) as as_of from {{ source('warehouse', 'fct_session') }}
),

base as (

    select
        u.user_id,
        u.n_sessions,
        u.n_buying_sessions,
        u.lifetime_revenue,
        u.conversion_rate,
        date_diff('day', u.last_seen, (select as_of from bounds)) as days_since_last_seen,
        date_diff('day', u.first_seen, u.last_seen)               as tenure_days

    from {{ source('warehouse', 'dim_user') }} u

),

scored as (

    select
        *,
        -- Recency / frequency / monetary quintiles. ntile is used rather than
        -- fixed cut-offs so the segmentation survives the ~26% conversion
        -- decline between October and November without silently re-labelling
        -- everyone.
        6 - ntile(5) over (order by days_since_last_seen)        as r_score,
        ntile(5) over (order by n_sessions)                      as f_score,
        ntile(5) over (order by lifetime_revenue)                as m_score

    from base

)

select
    user_id,
    n_sessions,
    n_buying_sessions,
    lifetime_revenue,
    conversion_rate,
    days_since_last_seen,
    tenure_days,
    r_score,
    f_score,
    m_score,
    r_score + f_score + m_score as rfm_total,

    case
        when n_buying_sessions = 0 and n_sessions >= 5 then 'browser_engaged'
        when n_buying_sessions = 0                     then 'browser_casual'
        when n_buying_sessions = 1                     then 'one_time_buyer'
        when r_score >= 4 and m_score >= 4             then 'champion'
        when r_score <= 2 and m_score >= 4             then 'at_risk_valuable'
        else                                                'repeat_buyer'
    end as segment

from scored
