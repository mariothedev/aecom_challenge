#!/usr/bin/env python3
"""Rebuild the three fictional practice PDFs. Requires reportlab.
Run from any directory: /usr/bin/python3 notes/generate_simple_data.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / 'simple_data'
WIDTH, HEIGHT = A4
LEFT, RIGHT = 42, 42
BODY_WIDTH = WIDTH - LEFT - RIGHT
STYLES = {
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=11, leading=16, spaceAfter=10),
    'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=14, leading=19),
    'sub': ParagraphStyle('sub', fontName='Helvetica-Bold', fontSize=11, leading=16),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=9, leading=13),
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=25, leading=32, alignment=TA_CENTER),
}


def p(text, style='body'):
    return Paragraph(escape(text), STYLES[style])


def table(rows, widths=None):
    result = Table([[p(str(cell), 'cell') for cell in row] for row in rows],
                   colWidths=widths or [BODY_WIDTH / len(rows[0])] * len(rows[0]))
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9edf0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#77838d')),
        ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d5dadd')),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    return result


class Document:
    def __init__(self, number, title, filename, revision='P01', date='2027-04-01'):
        self.number, self.title = number, title
        self.revision, self.date = revision, date
        self.c = canvas.Canvas(str(OUT / filename), pagesize=A4, invariant=1)
        self.c.setTitle(f'{number} {title}')
        self.c.setAuthor('Happy Paws Club (fictional)')
        self.c.setSubject('Happy Paws Dogs and Cats Club - HPC-2027-01 - RAG practice pack')
        self.page = 0

    def new_page(self):
        if self.page:
            self.c.showPage()
        self.page += 1
        c = self.c
        c.setFillColor(colors.HexColor('#36434c'))
        c.setFont('Helvetica', 8)
        c.drawString(LEFT, HEIGHT - 27, 'Happy Paws Club - Dogs and Cats Club - HPC-2027-01')
        c.drawRightString(WIDTH - RIGHT, HEIGHT - 41, f'{self.number} {self.title}')
        c.setStrokeColor(colors.HexColor('#c2c9ce'))
        c.line(LEFT, HEIGHT - 49, WIDTH - RIGHT, HEIGHT - 49)
        c.line(LEFT, 43, WIDTH - RIGHT, 43)
        c.drawString(LEFT, 28, 'Issued for Practice - Fictional example pack')
        c.drawRightString(WIDTH - RIGHT, 28, f'Page {self.page}')
        self.y = HEIGHT - 72

    def add(self, item):
        if isinstance(item, str):
            item = p(item)
        w, h = item.wrap(BODY_WIDTH, HEIGHT)
        if self.y - h < 62:
            raise ValueError(f'Overflow in document {self.number}, page {self.page}')
        item.drawOn(self.c, LEFT, self.y - h)
        self.y -= h + 11

    def heading(self, text):
        self.add(p(text, 'heading'))

    def sub(self, text):
        self.add(p(text, 'sub'))

    def cover(self):
        self.new_page()
        self.y -= 28
        self.add(p('Happy Paws Club', 'title'))
        self.add(p('Dogs and Cats Club', 'title'))
        self.add('Pack reference HPC-2027-01')
        self.y -= 28
        self.add(f'Document {self.number}')
        self.add(p(self.title, 'title'))
        self.y -= 25
        self.add(table([
            ['Revision', 'Date', 'Status', 'Prepared by', 'Checked by'],
            [self.revision, self.date, 'Issued for Practice', 'Club Team', 'Club Manager'],
        ]))
        self.add('This document is part of a fictional club pack. Read it with the other two documents and all issued addenda.')
        self.add('This is a simple reading and RAG exercise, not real animal-care advice. The club, people, rules and amounts are made up.')

    def save(self):
        self.c.save()
        print(f'{self.number}: {self.page} pages')


def rules():
    d = Document('01', 'Club Rules and Requirements', '01_Club_Rules_and_Requirements.pdf')
    d.cover()
    d.new_page()
    d.heading('1. Introduction')
    d.add('Happy Paws is a small daytime club for dogs and cats. The Club Team must prepare the rooms and provide the items in this pack. Dogs use Dog Room D-01. Cats use Cat Room C-01.')
    d.sub('1.1 Pack documents')
    d.add(table([['No.', 'Document', 'Revision'], ['01', 'Club Rules and Requirements', 'P01'], ['02', 'Rooms and Equipment Specification', 'P01'], ['03', 'Addenda and Query Responses', 'P02 (cumulative)']], [35, 365, BODY_WIDTH - 400]))
    d.sub('1.2 Definitions')
    d.add(table([['Term', 'Meaning'], ['Must', 'A rule that has to be followed.'], ['Should', 'A preference, not a strict rule.'], ['May', 'Something that is allowed but not required.'], ['Dog Room / Puppy Room', 'Two names for the same room, D-01.'], ['Cat Room / Kitten Room', 'Two names for the same room, C-01.'], ['Club Team', 'The people who set up and run the club.'], ['Owner', 'The person who brings and collects a pet.']]))
    d.heading('2. Which document wins?')
    d.add('If statements disagree, use this order: (1) numbered addenda in Document 03, with the later addendum winning; (2) Document 01; (3) Document 02. In the same document, a written rule wins over a table if they disagree. Still report the disagreement and show which statement wins.')
    d.add('Query responses in Document 03 Part C are information only. They do not change a rule unless a numbered addendum adopts the change. A newer query response does not win just because it is newer.')
    d.new_page()
    d.heading('3. Club dates and opening times')
    d.add(table([['Event', 'Date or time'], ['Pack issued', '1 April 2027'], ['Last day for setup questions', '20 April 2027'], ['Setup starts', '1 May 2027'], ['Setup complete', '15 May 2027'], ['Club opens', '16 May 2027'], ['Daily opening time', '09:00'], ['Daily closing time', '17:00']]))
    d.sub('3.1 Setup deadline')
    d.add('All setup work must finish by 15 May 2027. Work starts on 1 May 2027. The available setup period is 14 days, measured from the start of 1 May to the start of 15 May. Setup tasks must happen one after another, with no overlap.')
    d.heading('4. Pet and owner rules')
    d.sub('4.1 Dogs')
    d.add('At most 6 dogs may be in Dog Room D-01 at one time. Each dog must stay in D-01 during indoor play. A dog must have a lead attached when it moves through the entrance.')
    d.sub('4.2 Cats')
    d.add('At most 4 cats may be in Cat Room C-01 at one time. Each cat must stay in C-01 during indoor play. A cat must arrive and leave in a closed carrier.')
    d.sub('4.3 Collection')
    d.add('Owners must collect all pets by 17:00. The club has no overnight service.')
    d.new_page()
    d.heading('5. Rooms and safety')
    d.sub('5.1 Gates')
    d.add('Every indoor pet gate must be exactly 120 cm high. This includes gate G-D01 at Dog Room D-01 and gate G-C01 at Cat Room C-01. Keep both gates closed during play.')
    d.sub('5.2 Floors and beds')
    d.add('Both pet rooms must have washable floors. Each dog must have one bed and each cat must have one bed. Dog beds should be blue; other colours are allowed.')
    d.sub('5.3 Water bowls')
    d.add('Dog Room D-01 must have exactly 3 water bowls, each with a capacity of 2 litres. Cat Room C-01 must have exactly 2 water bowls, each with a capacity of 500 ml. These are bowl sizes, not daily drinking targets.')
    d.heading('6. Who supplies the items?')
    d.add('The Club Team must supply all beds, water bowls and cat scratching posts. Owners supply leads and carriers. The Club Team must supply exactly 2 scratching posts for Cat Room C-01.')
    d.heading('7. Cleaning')
    d.add('The Club Team must clean both pet rooms exactly twice each day: once before opening and once after closing. The first-aid box FA-01 must be checked before opening each day.')
    d.new_page()
    d.heading('8. Room data sheets')
    d.sub('D-01 Dog Room (also called Puppy Room)')
    d.add(table([['Item', 'Requirement'], ['Maximum dogs', '6'], ['Beds', '6 beds'], ['Water bowls', '3 bowls, 2 litres each'], ['Gate', 'G-D01, 120 cm high'], ['Floor', 'Washable']]))
    d.sub('C-01 Cat Room (also called Kitten Room)')
    d.add(table([['Item', 'Requirement'], ['Maximum cats', '4'], ['Beds', '4 beds'], ['Water bowls', '2 bowls, 500 ml each'], ['Gate', 'G-C01, 120 cm high'], ['Scratching posts', '2, supplied by the Club Team'], ['Floor', 'Washable']]))
    d.heading('9. Checklist and handover')
    d.add('The setup checklist must contain a row for every required safety item, including first-aid box FA-01. Before handover, the Club Team must check the rooms, gates and supplies, and give the Club Manager a signed checklist.')
    d.heading('10. Questions and changes')
    d.add('Ask setup questions in writing. The Club Manager records answers in Document 03 Part C. Only a numbered addendum changes this pack. Sign the acknowledgement in Document 03 Part D after reading the addenda.')
    d.save()


def specification():
    d = Document('02', 'Rooms and Equipment Specification', '02_Rooms_and_Equipment_Specification.pdf')
    d.cover()
    d.new_page()
    d.heading('1. General')
    d.sub('1.1 Scope')
    d.add('This document gives room details, equipment lists and the setup plan. Read it with Document 01 and Document 03. Use the document order in Document 01 Section 2 if statements disagree.')
    d.sub('1.2 Room names')
    d.add('Puppy Room means Dog Room D-01. Kitten Room means Cat Room C-01. These names do not create extra rooms or different age groups.')
    d.heading('2. Room specification')
    d.sub('2.1 Dog Room D-01')
    d.add('Dog Room D-01 must hold at most 8 dogs at one time. Provide a washable floor, 6 beds and water bowls as listed in Section 4.1. The dog beds must be washable. Their colour may be green.')
    d.sub('2.2 Cat Room C-01')
    d.add('Cat Room C-01 must hold at most 4 cats at one time. Provide a washable floor, 4 beds and water bowls as listed in Section 4.1. Owners must supply the 2 scratching posts for this room; the Club Team must not supply them.')
    d.heading('3. Gates')
    d.sub('3.1 Written gate rule')
    d.add('Both indoor pet gates must be exactly 120 cm high. Measure vertically from the floor to the top of the gate. Keep each gate closed during play.')
    d.new_page()
    d.sub('3.2 Gate schedule')
    d.add(table([['Gate tag', 'Location', 'Height'], ['G-D01', 'Dog Room D-01 entrance', '120 cm'], ['G-C01', 'Cat Room C-01 entrance', '90 cm']], [80, 300, BODY_WIDTH - 380]))
    d.heading('4. Equipment schedules')
    d.sub('4.1 Beds and water bowls')
    d.add('The values below are the required counts and bowl capacities, not examples or minimums. Bowl capacity is the amount one bowl can hold.')
    d.add(table([['Tag', 'Room', 'Item', 'Count', 'Capacity each'], ['BD-D', 'D-01', 'Washable bed', '6', '-'], ['BW-D', 'D-01', 'Water bowl', '3', '2 ml'], ['BD-C', 'C-01', 'Washable bed', '4', '-'], ['BW-C', 'C-01', 'Water bowl', '2', '0.5 litres']], [65, 60, 150, 55, BODY_WIDTH - 330]))
    d.sub('4.2 Other equipment')
    d.add(table([['Tag', 'Item', 'Count', 'Supplier'], ['SP-C', 'Cat scratching post in C-01', '2', 'Owners'], ['FA-01', 'First-aid box at reception', '1', 'Club Team']], [65, 230, 55, BODY_WIDTH - 350]))
    d.add('The first-aid box FA-01 must be ready before the club opens. Store it at reception, away from the pet rooms.')
    d.new_page()
    d.heading('5. Daily operation')
    d.sub('5.1 Cleaning')
    d.add('The Club Team must clean both pet rooms exactly once each day, after closing. No before-opening room clean is included.')
    d.sub('5.2 Opening and collection')
    d.add('The club opens at 09:00 and closes at 17:00. Owners must collect all pets by 17:00. Overnight stays are not included.')
    d.heading('6. Setup programme')
    d.add('Start setup on 1 May 2027. Carry out every task below in order, one after another, with no gaps and no overlap. A day means a full 24-hour period. This is the proposed setup plan; it does not change the deadline in Document 01 Section 3.1.')
    d.add(table([['Order', 'Task', 'Duration'], ['1', 'Clean the empty rooms', '5 days'], ['2', 'Fit gates and put in beds and bowls', '6 days'], ['3', 'Check safety and hand over', '5 days'], ['Total', 'All tasks', '16 days']], [60, 350, BODY_WIDTH - 410]))
    d.add('The proposed programme finishes at the start of 17 May 2027, 16 days after the start of 1 May 2027.')
    d.new_page()
    d.heading('7. Setup checklist')
    d.add('The table below is the complete setup checklist issued with this specification. Each required safety item must have its own row, as required by Document 01 Section 9.')
    d.add(table([['Item', 'Check', 'Signed'], ['D-01 floor', 'Washable and clean', '________'], ['C-01 floor', 'Washable and clean', '________'], ['G-D01', 'Height checked; gate closes', '________'], ['G-C01', 'Height checked; gate closes', '________'], ['BD-D and BD-C', 'All beds in place', '________'], ['BW-D and BW-C', 'All bowls in place', '________'], ['SP-C', '2 scratching posts in C-01', '________']]))
    d.heading('8. Handover')
    d.add('Give the Club Manager the signed setup checklist and a list of supplied items. Show the Club Manager how both gates close. Handover does not change the setup deadline.')
    d.heading('9. Care of equipment')
    d.add('Wash beds and bowls when needed. Replace broken equipment before it is used again. Keep owner leads and carriers with the correct pet.')
    d.save()


def addenda():
    d = Document('03', 'Addenda and Query Responses', '03_Addenda_and_Query_Responses.pdf', 'P02', '2027-04-18')
    d.cover()
    d.new_page()
    d.heading('Introduction')
    d.add('This document contains all changes issued so far. Parts A and B are binding numbered addenda. Part C contains informational query responses only. Part D records acknowledgement. Part E records visits. This P02 issue includes the earlier P01 content.')
    d.add('A later numbered addendum wins over an earlier one. An addendum changes only the topics and clauses it names. All other rules remain unchanged. Follow Document 01 Section 2.')
    d.add(table([['Issue', 'Date', 'Contents'], ['P01', '10 April 2027', 'Addendum No. 1; Q-01 and Q-02'], ['P02', '18 April 2027', 'Addendum No. 2; Q-03 to Q-05']]))
    d.heading('Part A - Addendum No. 1')
    d.add('Issued 10 April 2027 by the Club Manager.')
    d.sub('Item 1.1 - Maximum dogs')
    d.add('Change the maximum number of dogs in Dog Room D-01 to 5. This replaces the maximum of 6 in Document 01 Sections 4.1 and 8 and the maximum of 8 in Document 02 Section 2.1. Bed and bowl counts do not change.')
    d.sub('Item 1.2 - Dog water bowl capacity')
    d.add('In Document 02 Section 4.1, replace the capacity of each BW-D dog water bowl, 2 ml, with 2 litres. This agrees with Document 01 Section 5.3 and Section 8. Keep exactly 3 dog water bowls.')
    d.sub('Item 1.3 - Second club visit')
    d.add('A second club visit will be held on 19 April 2027 at 10:00. A visit is optional and does not change any setup date.')
    d.new_page()
    d.heading('Part B - Addendum No. 2')
    d.add('Issued 18 April 2027 by the Club Manager.')
    d.sub('Item 2.1 - Maximum dogs')
    d.add('Addendum No. 1 Item 1.1 is superseded. The maximum number of dogs in Dog Room D-01 is now 4. This replaces all earlier maximum-dog values in Documents 01 and 02 and Addendum No. 1. Bed and bowl counts do not change.')
    d.sub('Item 2.2 - Cat water bowl count')
    d.add('Cat Room C-01 must now have exactly 3 water bowls. This replaces the count of 2 in Document 01 Sections 5.3 and 8 and Document 02 Section 4.1. Each cat bowl still holds 500 ml (0.5 litres).')
    d.sub('Item 2.3 - Last day for questions')
    d.add('Change the last day for setup questions in Document 01 Section 3 from 20 April 2027 to 25 April 2027. The setup deadline and club opening date do not change.')
    d.sub('Item 2.4 - Items not changed')
    d.add('The gate heights, scratching-post supplier, cleaning frequency, setup task durations and setup checklist are not amended by this addendum. Query responses do not amend them either. Addendum No. 1 Item 1.2 remains in force.')
    d.new_page()
    d.heading('Part C - Queries and responses')
    d.add('These answers are information only. No answer in Part C has been adopted by an addendum. If an answer conflicts with a rule, follow the document order in Document 01 Section 2 and ask for clarification.')
    for tag, date, ref, question, answer in [
        ('Q-01', '8 April 2027', 'Document 01 Section 1.2 / Document 02 Section 1.2', 'Are Puppy Room and Dog Room different rooms?', 'No. Both names mean room D-01.'),
        ('Q-02', '9 April 2027', 'Document 01 Section 5.3 / Document 02 Section 4.1', 'Is 0.5 litres different from 500 ml for a cat bowl?', 'No. 0.5 litres and 500 ml are the same capacity.'),
        ('Q-03', '16 April 2027', 'Document 01 Section 7 / Document 02 Section 5.1', 'How often should we clean the rooms?', 'Once each day, after closing, is enough.'),
        ('Q-04', '17 April 2027', 'Document 01 Section 6 / Document 02 Section 2.2', 'Who supplies the cat scratching posts?', 'Owners supply both scratching posts.'),
        ('Q-05', '18 April 2027', 'Document 01 Section 3.1 / Document 02 Section 6', 'Does the 16-day setup plan extend the deadline?', 'No. A plan does not change the deadline. Ask the Club Manager to correct the plan.'),
    ]:
        d.sub(f'{tag} - {date}')
        d.add(p(f'Reference: {ref}\nQuestion: {question}\nResponse: {answer}'.replace('\n', ' '), 'cell'))
    d.new_page()
    d.heading('Part D - Acknowledgement of addenda')
    d.add('The Club Team must complete this record before handover.')
    d.add(table([['Addendum', 'Issue date', 'Received on', 'Initials'], ['No. 1', '10 April 2027', '________', '________'], ['No. 2', '18 April 2027', '________', '________']]))
    d.add('We confirm that we have read these addenda and used the current rules when preparing the rooms.')
    d.add('Signed: ____________________   Date: ____________________')
    d.heading('Part E - Record of club visits')
    d.add('A first club visit took place on 5 April 2027. The second visit is planned for 19 April 2027 at 10:00. Visits are optional. Attendance does not change the rules or replace reading the pack.')
    d.add(table([['Visit', 'Date', 'Record'], ['First', '5 April 2027', '3 Club Team members attended'], ['Second', '19 April 2027', 'Attendance to be recorded']]))
    d.save()


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rules()
    specification()
    addenda()
