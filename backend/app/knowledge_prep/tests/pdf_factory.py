"""PDF-uri minimale cu strat de text, pentru teste (fără reportlab)."""


def make_pdf(pages: list[list[str]]) -> bytes:
    """Fiecare pagină e o listă de rânduri ASCII; o listă goală dă o pagină fără text (ca una scanată)."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",  # /Pages, completat mai jos
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for lines in pages:
        ops = "".join(f"({line}) Tj T* " for line in lines)
        stream = f"BT /F1 11 Tf 14 TL 50 800 Td {ops}ET".encode()
        objects.append(
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
        )
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % len(objects)
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = (
        f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)
