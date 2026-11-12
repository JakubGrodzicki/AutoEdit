import openpyxl

from autoedit import tables


def test_split_workbook_one_file_per_sheet(tmp_path):
    wb = openpyxl.Workbook()
    wb.active.title = "Postac A"
    wb.active.append(["Nazwa Skryptu", "Postać", "Czy MC", "Treść kwestii"])
    wb.active.append(["S_01", "A", "NIE", "Cześć"])
    wb.create_sheet("A|B")
    wb.create_sheet("a<b")  # po zamianie znaków koliduje z "A|B"
    src = tmp_path / "aktor.xlsx"
    wb.save(src)

    files = tables.split_workbook(src, tmp_path / "out")

    assert [p.name for p in files] == ["Postac A.xlsx", "A_B.xlsx", "a_b_2.xlsx"]
    assert all(p.is_file() for p in files)
    lines = tables.load_lines(files[0])
    assert [ln.script for ln in lines] == ["S_01"]
    assert openpyxl.load_workbook(files[2]).sheetnames == ["a<b"]
