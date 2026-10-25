#!/usr/bin/env python3
"""Rebuild minimal_data with original document titles and retained section numbers.
Run: /usr/bin/python3 notes/generate_minimal_data.py
Requires reportlab. Content is fictional, not animal-care advice.
"""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'minimal_data'
WIDTH, HEIGHT = A4
MARGIN = 42
CONTENT_WIDTH = WIDTH - 2 * MARGIN
styles = {
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=14),
    'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=11, leading=15),
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=23, leading=30),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=9, leading=12),
}

# Retain original heading numbers. Omit irrelevant sections rather than renumbering.
# Each nested page is deliberately short; page 1 is always the original-style cover.
documents = [
    ('01_Instructions_to_Tenderers', 'Instructions to Tenderers', [
        [
            ('1. Introduction', 'Tiny Paws Club invites a team to prepare one dog room and one cat room.'),
            ('1.1 Tender documents', [
                ['No.', 'Document'],
                ['01', 'Instructions to Tenderers'],
                ['02', 'Particular Conditions of Contract'],
                ['03', "Employer's Requirements"],
                ['04', 'Architectural Specification'],
                ['05', 'Mechanical, Electrical and Public Health (MEP) Specification'],
                ['06', 'Drawing Register, Drawing Notes and Tender Programme'],
                ['07', 'Addenda and Tender Query Responses'],
            ]),
            ('1.2 Definitions', 'Employer means Tiny Paws Club. Contractor means the setup team. Must and shall mean required.'),
            ('1.3 Level and room naming convention', 'L00 means Ground Floor. L00-001 is the Dog Room. L00-002 is the Cat Room.'),
            ('2. Project description', 'Prepare the two existing rooms for daytime pet visits.'),
            ('3. Tender procedure and key dates', [
                ['Event', 'Date'], ['Setup starts', '1 May 2027'], ['Setup complete', '15 May 2027'],
            ]),
        ],
        [
            ('4. The proposed Contract', 'The Contractor follows Document 02.'),
            ('4.2 Contract period', 'Setup runs from 1 May to 15 May 2027.'),
            ('5. Order of precedence of documents', [
                ['Rank', 'Document'],
                ['1', 'Numbered addenda in Document 07; later addendum wins'],
                ['2', 'Document 02 - Particular Conditions of Contract'],
                ['3', "Document 03 - Employer's Requirements"],
                ['4', 'Documents 04 and 05 - Specifications'],
                ['5', 'Document 06 - Drawings and drawing notes'],
                ['6', 'Tables, if they conflict with written text in the same document'],
                ['7', 'Document 01 - Instructions to Tenderers'],
            ]),
            ('8. Tender queries and addenda', 'Only numbered addenda change requirements. A later addendum changes only the items it names. Query responses are information only, unless adopted by an addendum.'),
        ],
    ]),
    ('02_Particular_Conditions_of_Contract', 'Particular Conditions of Contract', [[
        ('Part A - Contract Data', [
            ['Item', 'Data'], ['Employer', 'Tiny Paws Club'], ['Contractor', 'The setup team'],
            ['Works', 'Prepare the dog and cat rooms'],
        ]),
        ('Part B - Amendments to the General Conditions', 'These are the contract rules for this practice pack.'),
        ('Clause 1 - Definitions', 'Completion means both rooms are ready to use.'),
        ('Clause 2 - Contract documents', 'Use the order of precedence in Document 01 Section 5.'),
        ('Clause 8 - Time', ''),
        ('8.1 Commencement', 'Start setup on 1 May 2027.'),
        ('8.2 Date for Completion', 'All setup must finish by 15 May 2027.'),
        ('8.3 Programme', 'The proposed programme does not change the completion deadline.'),
        ('Clause 11 - Completion and defects', ''),
        ('11.4 Handover documentation', 'Give the Employer a signed room checklist at Completion.'),
    ]]),
    ('03_Employers_Requirements', "Employer's Requirements", [[
        ('1. Purpose and status', 'This document states what the two rooms must provide.'),
        ('2. Scope of the Works', 'The Contractor prepares both rooms and supplies their gates and water bowls.'),
        ('5. The building', ''),
        ('5.6 Doors to escape routes', 'Cat Room gate D-G02 must be exactly 1,200 mm high, measured from floor to top.'),
        ('8. Room data sheets', [
            ['Room', 'Name', 'Maximum pets allowed'],
            ['L00-001', 'Dog Room', '6 dogs'],
            ['L00-002', 'Cat Room', '4 cats'],
        ]),
        ('10. Submittals and review', 'Submit the room checklist before handover.'),
    ]]),
    ('04_Architectural_Specification', 'Architectural Specification', [[
        ('1. General', ''),
        ('1.1 Scope', 'This specification covers room gates and floors.'),
        ('5. Doors and ironmongery', ''),
        ('5.1 General', 'In this club, the scheduled pet doors are gates. Heights are measured from floor to top.'),
        ('5.2 Door schedule', [
            ['Door', 'Location', 'Height (mm)'],
            ['D-G02', 'L00-002 Cat Room', '900'],
        ]),
        ('6. Floor, wall and ceiling finishes', 'Both rooms have washable floors.'),
        ('9. Handover', 'Show the Employer that the cat gate closes.'),
    ]]),
    ('05_MEP_Specification', 'Mechanical, Electrical and Public Health (MEP) Specification', [[
        ('1. General', ''),
        ('1.1 Scope', 'This specification covers water bowls and room ventilation.'),
        ('4. Public health', 'Supply one water bowl for each room. Each bowl holds 500 ml.'),
        ('6. Ventilation', ''),
        ('6.2 Design ventilation rates', 'The table sets the maximum pets allowed and the number of fans to supply.'),
        ('', [
            ['Space', 'Maximum pets allowed', 'Fans'],
            ['L00-001 Dog Room', '8 dogs', '1'],
            ['L00-002 Cat Room', '4 cats', '1'],
        ]),
        ('11. Testing, commissioning and handover', 'Check that both fans work before handover.'),
    ]]),
    ('06_Drawing_Notes_and_Tender_Programme', 'Drawing Register, Drawing Notes and Tender Programme', [[
        ('Part A - Drawing register', [
            ['Drawing', 'Title', 'Rev'], ['A-101', 'Ground floor plan L00', 'P01'],
        ]),
        ('Part B - Drawing notes', ''),
        ('A-101 Ground floor plan L00', 'L00-001 is the Dog Room. L00-002 is the Cat Room. Cat gate D-G02 is at the Cat Room entrance.'),
        ('Part C - Schedule of key figured dimensions', [
            ['Drawing', 'Room', 'Dimension', 'Value'],
            ['A-101', 'L00-002', 'D-G02 gate height', '900 mm'],
        ]),
        ('Part D - General notes applying to all drawings', 'Dimensions are in millimetres. Report disagreements with written requirements.'),
        ('Part E - Programme assumptions', 'Start setup on 1 May 2027.'),
        ('Part F - Tender programme', 'This is a proposed plan, not a new deadline.'),
        ('', [
            ['Activity', 'Start', 'Finish'],
            ['Prepare both rooms', '1 May 2027', '17 May 2027'],
        ]),
        ('Part G - Information documents', 'No extra information documents are issued.'),
    ]]),
    ('07_Addenda_and_Tender_Query_Responses', 'Addenda and Tender Query Responses', [
        [
            ('Introduction', 'This cumulative P02 issue contains both addenda. Later numbered addenda win. Other requirements remain unchanged.'),
            ('', [
                ['Issue', 'Date', 'Content'],
                ['P01', '10 April 2027', 'Addendum No. 1'],
                ['P02', '12 April 2027', 'Addendum No. 2 and TQ-01'],
            ]),
            ('Part A - Addendum No. 1', 'Issued 10 April 2027.'),
            ('Item 1.1 - Dog Room limit', 'Replace the dog limits in Document 03 Section 8 and Document 05 Section 6.2 with a maximum of 5 dogs in L00-001.'),
            ('Part B - Addendum No. 2', 'Issued 12 April 2027.'),
            ('Item 2.1 - Dog Room limit', 'Addendum No. 1 Item 1.1 is superseded. The maximum is now 4 dogs in L00-001. This replaces all earlier dog limits. No other requirement changes.'),
        ],
        [
            ('Part C - Tender queries and responses', 'Responses are information only. They do not change requirements unless adopted by a numbered addendum.'),
            ('TQ-01', [
                ['Field', 'Text'],
                ['Reference', 'Document 02 Clause 8.2 / Document 06 Part F'],
                ['Query', 'Does the proposed plan extend the setup deadline?'],
                ['Response', 'No. Completion is still due on 15 May 2027.'],
            ]),
            ('Part D - Acknowledgement of addenda', [
                ['Addendum', 'Issue date', 'Initials'],
                ['No. 1', '10 April 2027', '________'],
                ['No. 2', '12 April 2027', '________'],
            ]),
            ('Part E - Record of site visit attendance', 'No site visit is required for this small project.'),
        ],
    ]),
]


def paragraph(text, style='body'):
    return Paragraph(escape(text), styles[style])


def make_table(rows):
    cells = [[paragraph(cell, 'cell') for cell in row] for row in rows]
    widths = [CONTENT_WIDTH / len(rows[0])] * len(rows[0])
    if len(rows[0]) == 2:
        widths = [100, CONTENT_WIDTH - 100]
    result = Table(cells, colWidths=widths)
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9edf0')),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#a0a9b0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return result


def draw_block(pdf, block, y):
    _, block_height = block.wrap(CONTENT_WIDTH, HEIGHT)
    if y - block_height < 62:
        raise ValueError('Page overflow: shorten the content instead of shrinking the font.')
    block.drawOn(pdf, MARGIN, y - block_height)
    return y - block_height - 9


def page_frame(pdf, number, title, page):
    pdf.setFillColor(colors.HexColor('#36434c'))
    pdf.setFont('Helvetica', 8)
    pdf.drawString(MARGIN, HEIGHT - 25, 'Tiny Paws Club - Dogs and Cats Centre - TPC-2027-T01')
    pdf.drawString(MARGIN, HEIGHT - 40, f'{number} {title}')
    pdf.setStrokeColor(colors.HexColor('#c2c9ce'))
    pdf.line(MARGIN, HEIGHT - 48, WIDTH - MARGIN, HEIGHT - 48)
    pdf.line(MARGIN, 43, WIDTH - MARGIN, 43)
    pdf.drawString(MARGIN, 28, 'Issued for Tender - Fictional practice pack')
    pdf.drawRightString(WIDTH - MARGIN, 28, f'Page {page}')
    return HEIGHT - 66


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    for filename, title, pages in documents:
        number = filename[:2]
        revision = 'P02' if number == '07' else 'P01'
        date = '2027-04-12' if number == '07' else '2027-04-01'
        pdf = canvas.Canvas(str(OUT / f'{filename}.pdf'), pagesize=A4, invariant=1)
        pdf.setTitle(f'{number} {title}')
        pdf.setAuthor('Tiny Paws Club')
        pdf.setSubject('Dogs and Cats Centre - TPC-2027-T01')
        y = page_frame(pdf, number, title, 1) - 25
        for text, style in [
            ('Tiny Paws Club', 'title'),
            ('Dogs and Cats Centre', 'title'),
            ('Tender Reference TPC-2027-T01', 'body'),
            (f'Document {number}', 'heading'),
            (title, 'title'),
        ]:
            y = draw_block(pdf, paragraph(text, style), y)
        y -= 18
        y = draw_block(pdf, make_table([
            ['Revision', 'Date', 'Status', 'Prepared by', 'Checked by'],
            [revision, date, 'Issued for Tender' + (' - cumulative' if number == '07' else ''), 'Club Team', 'Club Manager'],
        ]), y)
        y = draw_block(pdf, paragraph('For tendering practice only; not for construction. This is a fictional dogs-and-cats example, not animal-care advice.'), y)
        for page_number, sections in enumerate(pages, 2):
            pdf.showPage()
            y = page_frame(pdf, number, title, page_number)
            for heading, text in sections:
                if heading:
                    y = draw_block(pdf, paragraph(heading, 'heading'), y)
                if text:
                    block = make_table(text) if isinstance(text, list) else paragraph(text)
                    y = draw_block(pdf, block, y)
        pdf.save()
        print(f'{filename}.pdf: {len(pages) + 1} pages')
