import pymupdf
import pytest

from pipeline.layout import analyze_layout


def test_extracts_table_and_body_from_requested_physical_page():
    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.insert_text((65, 125), "Unselected first page.")
        page = pdf.new_page()
        page.insert_text((65, 125), "Selected body text.")
        for x in (55, 155, 385):
            page.draw_line((x, 245), (x, 305))
        for y in (245, 275, 305):
            page.draw_line((55, y), (385, y))
        page.insert_text((65, 264), "Label")
        page.insert_text((165, 264), "Value")
        page.insert_text((65, 294), "Pressure")
        page.insert_text((165, 294), "0.1 MPa")

        layout = analyze_layout(pdf, 2, 2)
        block, = layout.body_blocks
        table, = layout.tables
        assert block.page_no == table.page_no == 2
        assert block.text == "Selected body text."
        assert list(page.get_textbox(pymupdf.Rect(block.bbox)).splitlines()) == [block.text]
        assert table.header == ["Label", "Value"]
        assert table.rows == [["Pressure", "0.1 MPa"]]
        assert 275 <= table.row_bboxes[0][1] < table.row_bboxes[0][3] <= 305


def test_analyze_layout_checks_physical_page_bounds():
    with pymupdf.open() as pdf:
        pdf.new_page()
        for span in [(0, 1), (1, 2), (2, 1)]:
            with pytest.raises(ValueError, match="范围"):
                analyze_layout(pdf, *span)
        assert analyze_layout(pdf).body_blocks == []
