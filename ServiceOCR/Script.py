import os
import sys
import warnings
from pdf2image import convert_from_path
import numpy as np
from doctr.models import ocr_predictor
from doctr.utils.visualization import visualize_page

warnings.filterwarnings("ignore")0

import google.generativeai as genai
import os

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel('gemini-1.5-flash')

from doctr.io import DocumentFile
from doctr.models import ocr_predictor




def pdf_to_text(pdf_path):
    doc = DocumentFile.from_pdf(pdf_path)
    predictor = ocr_predictor(pretrained=True)
    result = predictor(doc)
    texte = result.render()
    prompt = (
        "You are an OCR text organizer specializing in structuring financial documents. "
        "You will be provided with unstructured OCR outputs from various types of financial papers. "
        "Your task is to identify the type of document, and then organize the text according to a specific structure, keeping the fields the same but replacing the values with those extracted from the OCR output. "
        "If any field does not contain a value, mark it as 'None'. Below are examples of how the OCR output should be structured for different types of documents:\n\n"

        "Type 1: Acknowledgement Receipt\n"
        "Example OCR Output:\n"
        "Eritrean Strengthening Tax\n"
        "Administration Project\n"
        "DOMESTIC TAX DEPARTMENT\n\n"
        "Nature of request\n"
        "New registration Individual (non-business reasons)\n"
        "Request Code\n"
        "2022/03/14038\n"
        "Status of application\n"
        "Validated\n"
        "Effective Start Date\n"
        "10/04/2022\n"
        "Effective End Date\n"
        "10/04/2022\n"
        "Subject of request\n"
        "Registration\n"
        "Observation\n"
        "TIN\n"
        "501438410\n"
        "Name or Company\n"
        "M FAHN\n"
        "First name / Business COOKE\n"
        "Date : Monrovia, August 05, 2024\n"
        "Edited by Arabsoft Arabsoft\n"
        "Page 1 of 1\n\n"
        "...\n"
        "Required Structured Output:\n"
        "Acknowledgement Receipt\n"
        "Org : Eritrean Strengthening Tax Administration Project DOMESTIC TAX DEPARTMENT\n"
        "Nature of request : New registration Individual (non-business reasons)\n"
        "Request Code : 2022/03/14038\n"
        "Status of application : Validated\n"
        "Effective Start Date : 10/04/2022\n"
        "Effective End Date : 10/04/2022\n"
        "Subject of request : Registration\n"
        "Observation : None\n"
        "TIN : 501438410\n"
        "Name or Company : M FAHN\n"
        "First name / Business : COOKE\n"
        "Date : Monrovia, August 05, 2024\n"
        "Editor : Arabsoft\n\n"
        "...\n\n"
        "Type 2: Payment Receipt\n"
        "Example OCR Output:\n"
        "\n"
        "REPUBLIC OF LIBERIA\n"
        "Payment Receipt\n"
        "1304235/141281\n"
        "D\n"
        "sr\n"
        "Bill number\n"
        "144290\n"
        "DE\n"
        "Liberia Revenue. Authority\n"
        "Receipt Identifier :\n"
        "035\n"
        "DAVID WULU SLUWAR DAVID WULU\n"
        "Act code\n"
        "1268\n"
        "Act reference\n"
        "5674\n"
        "Act date\n"
        "12/20/2023\n"
        "LTD\n"
        "Kind of act\n"
        "LLA-Surveyors Licensing & Registration Board\n"
        "Visa and Stamp\n"
        "The following transactions are in good standing\n"
        "LICENSE FEES\n"
        "2023\n"
        "TAX DUE\n"
        "#100.00#\n"
        "Date of payment\n"
        "12/20/2023\n"
        "Received by\n"
        "Amount settled\n"
        "$100.00 USD\n"
        "cash payment $100.00 USD\n"
        "Amount in words\n"
        "HUNDRED USD\n"
        "Operation performed by\n"
        "Matthew Peters\n"
        "Monrovia, December 21, 2023\n"
        "Edited by LRA\n"
        "P.O.Box 101 MONROVIA - EMAIL:i info@mfdp.gov.lr\n"
        "Website: :\n"
        "GLRA\n"
        "htp/ievenuelmgov.r\n"
        "...\n"
        "Required Structured Output:\n"
        "\n"
        "Payment Receipt\n"
        "Org : REPUBLIC OF LIBERIA Liberia Revenue. Authority\n"
        "Payment Receipt : 1304235/141281\n"
        "Bill number : 144290\n"
        "Receipt Identifier : 035 DAVID WULU SLUWAR DAVID WULU\n"
        "Act Code : 1268\n"
        "Act reference : 5674\n"
        "Act date : 12/20/2023\n"
        "Kind of Act : LLA-Surveyors Licensing & Registration Board\n"
        "LICENSE FEES : 2023\n"
        "TAX DUE : #100.00#\n"
        "Date of payment : 12/20/2023\n"
        "Received by : cash payment $100.00 USD\n"
        "Amount settled : $100.00 USD\n"
        "Amount in words : HUNDRED USD\n"
        "Operation performed by : Matthew Peters\n"
        "Date : Monrovia, December 21, 2023\n"
        "Editor : LRA\n"
        "Address : P.O.Box 101 MONROVIA\n"
        "Email : info@mfdp.gov.lr\n"
        "website : http/revenue.lra.gov.lr\n"
        "...\n\n"

        "Type 3: Receipt of Tax Return\n"
        "Example OCR Output:\n"
        "\n"
        "Eritrean Strengthening Tax\n"
        "Administration Project\n"
        "DOMESTIC TAX DEPARTMENT\n"
        "Receipt of tax return\n"
        "No: 1160575\n"
        "Taxpayer Identification\n"
        "TIN\n"
        ": 500742636T\n"
        "Name / Company name : THE RESCUE SOULS & CHRIST MINISTRIES INTERNATIONAL, INC.\n"
        "Date of entry into operation : 08/01/2012\n"
        "Activity\n"
        ": 9491 ACTIVITIES OF RELIGIOUS ORGANIZATIONS\n"
        "Business register\n"
        ":\n"
        "Dated:\n"
        "Legal form : FOR NOT PROFIT CORPORATION\n"
        "Tax Division : TEAM FOUR - GNFPD\n"
        "Tax return\n"
        "Code : 1160575\n"
        "Date of Tax return : 06/21/2024\n"
        "Period : ANNUAL 2024\n"
        "Kind of tax : Real Property Tax\n"
        "1/2\n"
        "Edited by ErITAS\n"
        "BP 912 BANGUICENTRAFRICAINE - EMAIL : info@mfdp.gov.lr\n"
        "Website : htp/revenuelmagovylr\n"
        "\n"
        "Eritrean Strengthening Tax\n"
        "Administration Project\n"
        "DOMESTIC TAX DEPARTMENT\n"
        "Receipt of tax return\n"
        "Principal Amount USD : $37,500.00 USD\n"
        "Penalties : $0.00 USD\n"
        "Nil Return : NO\n"
        "Credit : NO\n"
        "Amount of credit : $0.00 USD\n"
        "Total : $37,500.00 USD\n"
        "B. DECLARED\n"
        "Group\n"
        "Line\n"
        "A. PROPERTY TYPE\n"
        "Old Estimated value TAX. AMOUNT\n"
        "Tax Rate\n"
        "VALUE\n"
        "Property Declared\n"
        "24\n"
        "15000000\n"
        "0\n"
        "37500\n"
        ".25\n"
        "Value\n"
        "Address :\n"
        "Amount on hand to pay : $37,500.00 USD\n"
        "VISA\n"
        "2/2\n"
        "Edited by ErITAS\n"
        "BP 912 BANGUICENTRAFRICAINE - EMAIL : info@mfdp.gov.lr\n"
        "Website : htp./revenuelmagoyir\n"
        "...\n"
        "Required Structured Output:\n"
        "\n"
        "Receipt of tax return\n"
        "\n"
        "Org : Eritrean Strengthening Tax Administration Project DOMESTIC TAX DEPARTMENT\n"
        "No: 1160575\n"
        "TIN : 500742636T\n"
        "Name / Company name : THE RESCUE SOULS & CHRIST MINISTRIES INTERNATIONAL, INC.\n"
        "Date of entry into operation : 08/01/2012\n"
        "Activity : 9491 ACTIVITIES OF RELIGIOUS ORGANIZATIONS\n"
        "Business register : None\n"
        "Dated : None\n"
        "Legal form : FOR NOT PROFIT CORPORATION\n"
        "Tax Division : TEAM FOUR - GNFPD\n"
        "Code : 1160575\n"
        "Date of Tax return : 06/21/2024\n"
        "Period : ANNUAL 2024\n"
        "Kind of tax : Real Property Tax\n"
        "Principal Amount USD : $37,500.00 USD\n"
        "Penalties : $0.00 USD\n"
        "Nil Return : NO\n"
        "Credit : NO\n"
        "Amount of credit : $0.00 USD\n"
        "Total : $37,500.00 USD\n"
        "Group : Property Declared Value\n"
        "Line : .\n"
        "A. Property Type : 24\n"
        "B. Declared Value : 15000000\n"
        "Old Estimated Value : 0\n"
        "Tax Amount : 37500\n"
        "Tax Rate : .25\n"
        "Address : None\n"
        "Amount on hand to pay : $37,500.00 USD\n"
        "Payment Method : VISA\n"
        "Editor : ErITAS\n"
        "Address : BP 912 BANGUICENTRAFRICAINE\n"
        "Email : info@mfdp.gov.lr\n"
        "Website : http://revenue.lra.gov.lr\n"
        "Type 4: Banking Payment (JSON Output)\n"
        "Example OCR Output:\n"
        "\n"
        "Eritrean Strengthening Tax Administration\n"
        "Project\n"
        "DOMESTIC TAX DEPARTMENT\n"
        "Monrovia, August 05, 2024\n"
        "BANKING PAYMENT\n"
        "Document No 73801\n"
        "Currency: LIBERIAN DOLLAR (LRD)\n"
        "TIN TAX PAYER NAME\n"
        "ADRESS\n"
        "BRANCH\n"
        "TAXPAYER\n"
        "CATEGORY\n"
        "LARGE TAX\n"
        "501527788\n"
        "Arabsoft\n"
        "DIVSION\n"
        "Tax account\n"
        "Period\n"
        "Filing Date\n"
        "Filing due Date\n"
        "54677715471\n"
        "february 2021\n"
        "10/14/2022\n"
        "03/22/2021\n"
        "Code\n"
        "Tax / Fee Type\n"
        "Tax/Fee due\n"
        "Penalty\n"
        "Interest\n"
        "Total\n"
        "33335 EXCISE TAX\n"
        "$5,736,375,000.00\n"
        "$2,868,187,500.00\n"
        "$2,151,140,625.00\n"
        "$10,755,703,125.00\n"
        "11393\n"
        "$139,562.50\n"
        "$69,861.00\n"
        "$52,316.00\n"
        "$261,739.50\n"
        "37\n"
        "11512\n"
        "$72,562.50\n"
        "$36,281.25\n"
        "$27,251.25\n"
        "$136,095.00\n"
        "22\n"
        "11381\n"
        "$139,562.50\n"
        "$69,861.00\n"
        "$52,316.00\n"
        "$261,739.50\n"
        "21\n"
        "Amount on hand to pay :\n"
        "10,756,362.699.00 LRD\n"
        "Comment : Nothing\n"
        "NB: The total amount of the Bill must be paid in full.\n"
        "\n"
        "Section of the Bank\n"
        "Document No :73801\n"
        "Tax Period\n"
        "february 2021\n"
        "Tax Center\n"
        "TIN :\n"
        "501527788\n"
        "LARGE TAX DIVSION\n"
        "Tax/fee type :\n"
        "EXCISE TAX\n"
        "Amount paid:\n"
        "Cashier :\n"
        "Signature :\n"
        "\n"
        "Required Structured Output (JSON):\n"
        "\n"
        "{\n"
        '  "infos_document": {\n'
        '    "nom_du_projet": "Eritrean Strengthening Tax Administration Project",\n'
        '    "departement": "DOMESTIC TAX DEPARTMENT",\n'
        '    "emplacement": "Monrovia",\n'
        '    "date": "2024-08-05",\n'
        '    "numero_de_document": 73801,\n'
        '    "devise": "LIBERIAN DOLLAR (LRD)"\n'
        "  },\n"
        '  "infos_contribuable": {\n'
        '    "tin": "501527788",\n'
        '    "nom": "Arabsoft",\n'
        '    "adresse": "None",\n'
        '    "succursale": "None",\n'
        '    "categorie": "LARGE TAX DIVISION"\n'
        "  },\n"
        '  "Comptes_fiscaux":  [\n'
        "   {\n"
        '    "Numero": "54677715471",\n'
        '    "periode_comptable_fiscale": "February 2021",\n'
        '    "date_de_depot": "2022-10-14",\n'  # Corrected date format to be "YYYY-MM-DD"
        '    "date_limite_de_depot": "2021-03-22"\n'
        "    }\n"
        "  ],\n"
        '  "detail_des_frais_fiscaux": [\n'
        "    {\n"
        '      "code": 33335,\n'
        '      "type": "EXCISE TAX",\n'
        '      "frais_fiscaux_dus": 5736375000.00,\n'
        '      "penalite": 2868187500.00,\n'
        '      "interet": 2151140625.00,\n'
        '      "total": 10755703125.00\n'
        "    },\n"
        "    {\n"
        '      "code": 11393,\n'
        '      "type": "None",\n'
        '      "frais_fiscaux_dus": 139562.50,\n'
        '      "penalite": 69861.00,\n'
        '      "interet": 52316.00,\n'
        '      "total": 261739.50\n'
        "    },\n"
        "    {\n"
        '      "code": 11512,\n'
        '      "type": "None",\n'
        '      "frais_fiscaux_dus": 72562.50,\n'
        '      "penalite": 36281.25,\n'
        '      "interet": 27251.25,\n'
        '      "total": 136095.00\n'
        "    },\n"
        "    {\n"
        '      "code": 11381,\n'
        '      "type": "None",\n'
        '      "frais_fiscaux_dus": 139562.50,\n'
        '      "penalite": 69861.00,\n'
        '      "interet": 52316.00,\n'
        '      "total": 261739.50\n'
        "    }\n"
        "  ],\n"
        '  "infos_paiement": {\n'
        '    "montant_disponible_pour_payer": 10756362699.00,\n'
        '    "commentaire": "Nothing"\n'
        "  }\n"
        "}\n"
        "Please organize the following OCR output according to the examples provided above. "
        "For 'Acknowledgement Receipt', 'Payment Receipt', and 'Receipt of Tax Return', return a structured text format. "
        "For 'Banking Payment', return a structured JSON output.\n\n"
        f"{texte}"
    )
    response = model.generate_content(prompt)
    return response.text





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("le script python attendre 3 argument : Script.py <input_pdf_path> [output_path] mais il ya un seul argument")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        #kan mathamach output path bich ykoun howa bidou el input path
        output_path = os.path.splitext(pdf_path)[0] + ".txt"
        #Fonction os.path.splitext hathi traja3 liste faha root awil haja wi mba3id el extension par exemple : fi path hatha "/path/to/file/document.pdf" el root howa "/path/to/file/document" wi extension hiya pdf

    text = pdf_to_text(pdf_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(text)
