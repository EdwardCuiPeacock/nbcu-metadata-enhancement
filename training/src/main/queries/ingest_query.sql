SELECT 
  CONCAT(program_title, '. ', synopsis) as synopsis,
  tags
FROM `{{ project }}.{{ dataset }}.{{ table }}`
{% if limit -%}
   LIMIT {{ limit }}
{% endif %}