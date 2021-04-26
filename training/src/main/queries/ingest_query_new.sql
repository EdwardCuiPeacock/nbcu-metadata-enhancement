
WITH raw_programs AS (
    SELECT program_title, program_val AS program_id, program_longsynopsis, program_type, program_language
    FROM `{{ GOOGLE_CLOUD_PROJECT }}.content_metadata.merlin_program`
    WHERE program_longsynopsis IS NOT NULL
    AND program_seriesid IS NULL -- reducing almost 5/6 of the data
    AND program_language IN ("eng", "spa")
    AND program_type IN ("Episode", "Movie")
    AND program_title NOT IN ("OnDemand Movie", "Movie")
),

program_processed AS (
    SELECT program_title, program_id, program_longsynopsis,
        ARRAY_AGG(DISTINCT partial_tags_array IGNORE NULLS) AS partial_tags -- combine (type, language) into an array
    FROM raw_programs,
        UNNEST([program_type, program_language]) partial_tags_array
    GROUP BY program_title, program_id, program_longsynopsis, program_type, program_language
),

age_tags AS (
    SELECT DISTINCT program_id,
    CASE tag_value
        WHEN 'little kids (ages 5-7)' THEN 'kids (ages 5-9)'
        WHEN 'big kids (ages 8-9)' THEN 'kids (ages 5-9)'
        ELSE tag_value
    END tag_value
    FROM `{{ GOOGLE_CLOUD_PROJECT }}.content_metadata.merlin_tags`
    WHERE tag_value IN ("not for kids", "preschoolers (ages 2-4)", "little kids (ages 5-7)",
            "big kids (ages 8-9)", "tweens (ages 10-12)", "teens (ages 13-14)", "older teens (ages 15+)")
),

raw_tags AS (
    SELECT DISTINCT program_id, 
    CASE tag_value
        WHEN 'Docudrama' THEN 'Documentary drama'
        WHEN 'Historical drama' THEN 'History drama'
        WHEN 'Romantic comedy' THEN 'Romance comedy'
        ELSE tag_value
    END AS tag_value
    FROM `{{ GOOGLE_CLOUD_PROJECT }}.content_metadata.merlin_tags`
    WHERE tag_type = "Genre"
),

tag_counter AS (SELECT tag_value, COUNT(tag_value) AS counter
    FROM raw_tags
    GROUP BY tag_value
    ORDER BY counter DESC
    LIMIT 100
),

filtered_tag AS (
    SELECT x.program_id, x.tag_value
    FROM raw_tags x
    JOIN tag_counter y
    ON x.tag_value = y.tag_value
    
    UNION ALL
    
    SELECT program_id, tag_value
    FROM age_tags
),

tags_processed AS (
    SELECT program_id, ARRAY_AGG(DISTINCT tag_value) AS tag_value
    FROM filtered_tag
    GROUP BY program_id
),

program_tags_map AS (
SELECT a.program_title, a.program_id, 
    program_longsynopsis,
    ARRAY_CONCAT(a.partial_tags, b.tag_value) AS tags
FROM program_processed a
LEFT JOIN tags_processed b
ON a.program_id = b.program_id
WHERE ARRAY_LENGTH(a.partial_tags) + ARRAY_LENGTH(b.tag_value) > 0
),

program_tags_agg AS (SELECT program_title, program_id, program_longsynopsis, -- remove repeated synopsis
    ARRAY_CONCAT_AGG(tags) AS tags
FROM program_tags_map
GROUP BY program_longsynopsis, program_title, program_id)

SELECT program_title, program_id, program_longsynopsis AS synopsis, 
    (SELECT ARRAY_AGG(DISTINCT t) FROM UNNEST(tags) t) AS tags -- deduplicate tags
FROM program_tags_agg
{% if TEST_LIMIT -%}
   LIMIT {{ TEST_LIMIT }}
{% endif %}