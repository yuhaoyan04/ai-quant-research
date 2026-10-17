from quant_research.report_pdf import markdown_blocks


def test_markdown_blocks_parse_heading_table_and_bullet() -> None:
    blocks = markdown_blocks("""# Title

- item

| A | B |
|---|---|
| 1 | 2 |
""")
    assert blocks[0] == ("h1", "Title")
    assert blocks[1] == ("bullet", "item")
    assert blocks[2] == ("table", [["A", "B"], ["1", "2"]])
