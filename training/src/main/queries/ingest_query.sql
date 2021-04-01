SELECT 
  CONCAT(program_title, '. ', program_longsynopsis) as synopsis,
  tags
FROM `{{ project }}.{{ dataset }}.{{ table }}`
{% if limit -%}
   LIMIT {{ limit }}
{% endif %}