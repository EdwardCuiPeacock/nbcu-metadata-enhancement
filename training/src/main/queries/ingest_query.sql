SELECT 
  ARRAY_TO_STRING(ARRAY(
    SELECT * 
        FROM UNNEST(SPLIT(program_longsynopsis, " ")) LIMIT {{ TOKEN_LIMIT }}), " ") as synopsis,
  tags, tokens
FROM `{{ GOOGLE_CLOUD_PROJECT }}.metadata_enhancement.meta_synopsis_100tag_edc_dev`
{% if TEST_LIMIT -%}
   LIMIT {{ TEST_LIMIT }}
{% endif %}