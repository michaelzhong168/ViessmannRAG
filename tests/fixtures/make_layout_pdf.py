"""Generate the deterministic synthetic PDF used for G03 acceptance."""

from pathlib import Path

import pymupdf

DEFAULT_OUT = Path(__file__).with_name("layout.pdf")


def _draw_grid(page, x, y, values):
    xs = (x, x + 70, x + 140)
    ys = tuple(y + 30 * row for row in range(len(values) + 1))
    for edge in xs:
        page.draw_line((edge, ys[0]), (edge, ys[-1]))
    for edge in ys:
        page.draw_line((xs[0], edge), (xs[-1], edge))
    for row, cells in enumerate(values):
        for col, text in enumerate(cells):
            if text:
                page.insert_text(
                    (xs[col] + 6, ys[row] + 19), text, fontsize=10
                )


def build(out_path=DEFAULT_OUT):
    path = Path(out_path)
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=500, height=700)
        # Deliberately insert these out of reading order.
        for y, text, size in (
            (665, "Table of Contents 69", 10),
            (185, "Second body", 10),
            (105, "Title A", 24),
            (20, "Running header", 10),
            (72, "Opening body", 10),
            (155, "Title B", 20),
            (130, "Pressure 0.15 bar", 10),
            (225, "Title C", 16),
            (260, "Title D", 13),
            (290, "Title E", 13.04),
        ):
            page.insert_text((50, y), text, fontsize=size)
        _draw_grid(
            page, 50, 340, (("Fault Code", "Cause"), ("F4", "Low"), ("E5", ""))
        )

        page = pdf.new_page(width=300, height=500)
        page.insert_text((35, 470), "Rotated body", fontsize=10, color=(1, 0, 0))
        _draw_grid(
            page, 70, 310, (("State", "Value"), ("On", "1"), ("Off", "0"))
        )
        page.set_rotation(90)
        pdf.set_metadata({"title": "G03 layout fixture", "creator": Path(__file__).name})
        pdf.save(path, no_new_id=True)
    return path


if __name__ == "__main__":
    print(f"wrote {build()}")
