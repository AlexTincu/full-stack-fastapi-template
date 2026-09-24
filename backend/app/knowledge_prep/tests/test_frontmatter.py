from app.knowledge_prep.frontmatter import (
    FrontMatter,
    parse,
    render,
    slugify,
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


def test_render_then_parse_roundtrips_metadata_and_body():
    fm = FrontMatter(
        title="Rostul suferinței",
        slug="rostul-suferintei",
        type="transcriere",
        author=None,
        source_file="files/rostul suferintei.ogg",
    )
    body = "# Titlu\n\nText cu ș și ț.\n"

    parsed_fm, parsed_body = parse(render(fm, body))

    assert parsed_fm == fm
    assert parsed_fm.status == "draft"
    assert parsed_body == body


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
