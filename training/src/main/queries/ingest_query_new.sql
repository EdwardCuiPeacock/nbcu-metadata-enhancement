WITH raw_programs AS (
    SELECT program_val AS program_id,
        program_longsynopsis,
        program_type,
        program_language
    FROM `{{ GOOGLE_CLOUD_PROJECT }}.content_metadata.merlin_program`
    WHERE program_longsynopsis IS NOT NULL
        AND program_seriesid IS NULL -- reducing almost 5/6 of the data
        AND program_language IN ("eng", "spa")
        AND program_type IN ("Episode", "SeriesMaster", "Movie")
        AND program_title NOT LIKE "OnDemand Movie"
),
program_processed AS (
    SELECT program_id,
        program_longsynopsis,
        ARRAY_AGG(DISTINCT partial_tags_array IGNORE NULLS) AS partial_tags -- combine (type, language) into an array
    FROM raw_programs,
        UNNEST([program_type, program_language]) partial_tags_array
    GROUP BY program_id,
        program_longsynopsis,
        program_type,
        program_language
),
raw_tags AS (
    SELECT DISTINCT program_id,
        CASE
            tag_value
            WHEN 'little kids (ages 5-7)' THEN 'kids (ages 5-9)'
            WHEN 'big kids (ages 8-9)' THEN 'kids (ages 5-9)'
            WHEN 'Docudrama' THEN 'Documentary drama'
            WHEN 'Historical drama' THEN 'History drama'
            WHEN 'Romantic comedy' THEN 'Romance comedy'
            ELSE tag_value
        END AS tag_value
    FROM `{{ GOOGLE_CLOUD_PROJECT }}.content_metadata.merlin_tags`
    WHERE tag_type in ("Genre", "KidsTheme")
),
tags_processed AS (
    SELECT program_id,
        ARRAY_AGG(DISTINCT tag_value) AS tag_value
    FROM raw_tags
    GROUP BY program_id
),
program_tags_map AS (
    SELECT a.program_id,
        ARRAY_TO_STRING(
            ARRAY(
                SELECT *
                FROM UNNEST(SPLIT(a.program_longsynopsis, " "))
                LIMIT 256
            ), " "
        ) AS program_longsynopsis,
        ARRAY_CONCAT(a.partial_tags, b.tag_value) AS tags
    FROM program_processed a
        LEFT JOIN tags_processed b ON a.program_id = b.program_id
    WHERE ARRAY_LENGTH(a.partial_tags) + ARRAY_LENGTH(b.tag_value) > 0
),
program_tags_agg AS (
    SELECT program_longsynopsis,
        -- remove repeated synopsis
        ARRAY_CONCAT_AGG(tags) AS tags
    FROM program_tags_map
    GROUP BY program_longsynopsis
)
SELECT program_longsynopsis AS synopsis,
    (
        SELECT ARRAY_AGG(DISTINCT t)
        FROM UNNEST(tags) t
    ) AS tags -- deduplicate tags
FROM program_tags_agg 
{% if TEST_LIMIT -%}
   LIMIT {{ TEST_LIMIT }}
{% endif %}