SELECT *
FROM `{{ GOOGLE_CLOUD_PROJECT }}.metadata_enhancement.meta_synopsis_tag_edc_dev`
{% if TEST_LIMIT -%}
   LIMIT {{ TEST_LIMIT }}
{% endif %}