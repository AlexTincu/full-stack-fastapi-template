from dataclasses import asdict

import pytest

from app.knowledge_prep.frontmatter import (
    FrontMatter,
    render,
    slugify,
    split_frontmatter,
    strip_frontmatter,
    title_from_filename,
)


def test_slugify_strips_romanian_diacritics():
    assert (
        slugify("Sănătate din farmacia Domnului: ceaiuri și tincturi")
        == "sanatate-din-farmacia-domnului-ceaiuri-si-tincturi"
    )


def test_slugify_handles_cedilla_variants_and_separators():
    assert (
        slugify("Regim DRI - Boli grave ş.a. (ţesuturi)")
        == "regim-dri-boli-grave-s-a-tesuturi"
    )


def test_title_from_filename_replaces_underscores_and_hyphens():
    assert (
        title_from_filename("Maria_Treben_Sanatate_din_farmacia-domnului")
        == "Maria Treben Sanatate din farmacia domnului"
    )


def test_title_from_filename_keeps_spaced_dash_as_separator():
    assert (
        title_from_filename("Regim DRI - Boli hepatice") == "Regim DRI - Boli hepatice"
    )


def test_render_then_split_roundtrips_metadata_and_body():
    fm = FrontMatter(
        title="Rostul suferinței",
        slug="rostul-suferintei",
        type="transcriere",
        author=None,
        source_file="rostul suferintei.ogg",
    )
    body = "# Titlu\n\nText cu ș și ț.\n"

    meta, parsed_body = split_frontmatter(render(fm, body))

    assert meta == asdict(fm)
    assert "status" not in meta
    assert parsed_body == body


def test_split_frontmatter_without_frontmatter_returns_whole_text():
    text = "<!-- bucata 020 -->\n\n## Nuci\n"
    assert split_frontmatter(text) == ({}, text)


def test_split_frontmatter_keeps_unknown_keys_like_legacy_status():
    meta, body = split_frontmatter("---\ntitle: X\nstatus: reviewed\n---\n\ncorp\n")
    assert meta == {"title": "X", "status": "reviewed"}
    assert body == "corp\n"


def test_split_frontmatter_rejects_unclosed_block():
    with pytest.raises(ValueError, match="neînchis"):
        split_frontmatter("---\ntitle: X\ncorp\n")


def test_strip_frontmatter_ignores_metadata():
    fm = FrontMatter(title="A", slug="a", type="carte", author=None)
    assert strip_frontmatter(render(fm, "corp\n")) == "corp\n"
    assert strip_frontmatter("fără front-matter\n") == "fără front-matter\n"


def test_render_puts_frontmatter_block_first_and_keeps_diacritics_readable():
    text = render(
        FrontMatter(
            title="Păpădia",
            slug="papadia",
            type="nota",
            author="Maria Treben",
            source_file="files/x.pdf",
        ),
        "corp",
    )

    assert text.startswith("---\n")
    assert "title: Păpădia\n" in text
    assert text.endswith("---\n\ncorp")
