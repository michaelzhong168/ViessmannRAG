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


def test_rotated_bounding_boxes_follow_display_coordinates():
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=300, height=500)
        page.insert_text((65, 120), "Rotated text.")
        source_box = pymupdf.Rect(page.get_text("blocks")[0][:4])
        for x in (70, 130, 210):
            page.draw_line((x, 240), (x, 300))
        for y in (240, 270, 300):
            page.draw_line((70, y), (210, y))
        page.insert_text((80, 260), "Label")
        page.insert_text((140, 260), "Value")
        page.insert_text((80, 290), "Pressure")
        page.insert_text((140, 290), "0.1 MPa")
        page.set_rotation(90)
        layout = analyze_layout(pdf)

        body, = layout.body_blocks
        table, = layout.tables
        assert page.rotation == 90
        assert body.bbox == pytest.approx([500 - source_box.y1, source_box.x0,
                                           500 - source_box.y0, source_box.x1])
        assert table.bbox == pytest.approx([200, 70, 260, 210])
        assert table.row_bboxes[0] == pytest.approx([200, 70, 230, 210])
        assert table.rows == [["Pressure", "0.1 MPa"]]


def test_bookmarks_chapter_levels_and_margin_filtering():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for y, value, size in (
            (30, "Running header.", 10),
            (115, "Main title", 24),
            (175, "Chapter B", 18),
            (235, "Chapter C", 15),
            (295, "Chapter D", 13),
            (355, "Chapter D continues", 13.04),
            (430, "A longer body paragraph about operation.", 10),
            (500, "1. Press OK", 15),
        ):
            page.insert_text((50, y), value, fontsize=size)
        next_page = pdf.new_page()
        next_page.insert_text((50, 120), "A new page continues the chapter.", fontsize=10)
        next_page.insert_text((50, 190), "Bookmarked title", fontsize=10)
        next_page.insert_text((50, 260), "A paragraph under the bookmark.", fontsize=10)
        next_page.insert_text((50, 824), "Running footer.", fontsize=10)
        pdf.set_toc([[1, "Main title", 1], [1, "Bookmarked title", 2]])

        layout = analyze_layout(pdf)
        assert [(heading.text, heading.level, heading.page_no) for heading in layout.headings] == [
            ("Main title", 1, 1), ("Chapter B", 2, 1), ("Chapter C", 3, 1),
            ("Chapter D", 3, 1), ("Chapter D continues", 3, 1), ("Bookmarked title", 3, 2),
        ]
        assert {block.text: block.chapter for block in layout.body_blocks} == {
            "A longer body paragraph about operation.": "Chapter D continues",
            "1. Press OK": "Chapter D continues",
            "A new page continues the chapter.": "Chapter D continues",
            "A paragraph under the bookmark.": "Bookmarked title",
        }
        assert layout.header_footer_removed == 2
        unfiltered = analyze_layout(pdf, top_ratio=0, bottom_ratio=0)
        assert unfiltered.header_footer_removed == 0
        assert {"Running header.", "Running footer."} <= {block.text for block in unfiltered.body_blocks}
