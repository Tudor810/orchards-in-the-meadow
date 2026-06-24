from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import os

# --- Configurarea Culorilor ---
COLOR_TEXT = RGBColor(51, 51, 51)       # Charcoal
COLOR_BLUE = RGBColor(41, 128, 185)     # Deep Blue 
COLOR_GREEN = RGBColor(39, 174, 96)     # Forest Green
COLOR_ORANGE = RGBColor(211, 84, 0)     # Alert Orange
COLOR_RED = RGBColor(192, 57, 43)       # Red

def apply_text_style(text_frame, font_name='Calibri', font_size=18, color=COLOR_TEXT):
    for paragraph in text_frame.paragraphs:
        paragraph.font.name = font_name
        if paragraph.level == 0:
            paragraph.font.size = Pt(font_size)
        else:
            paragraph.font.size = Pt(font_size - 3)
        paragraph.font.color.rgb = color

def create_presentation():
    prs = Presentation()
    
    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Tratarea fenomenului extrem de Class Imbalance în datele satelitare de tip Time-Series"
    slide.shapes.title.text_frame.paragraphs[0].font.size = Pt(32)
    slide.shapes.title.text_frame.paragraphs[0].font.color.rgb = COLOR_BLUE
    slide.shapes.title.text_frame.paragraphs[0].font.bold = True

    subtitle = slide.placeholders[1]
    subtitle.text = "O abordare bazată pe Dynamic Mixture of Experts și Late-Fusion Meta-Classifiers\n\nAgricultureAi Project"
    apply_text_style(subtitle.text_frame, font_size=20)
    subtitle.text_frame.paragraphs[2].font.color.rgb = COLOR_GREEN
    subtitle.text_frame.paragraphs[2].font.bold = True

    # ==========================================
    # SLIDE 2: Problema și Motivația
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "1. Problema și Motivația"
    tf = slide.placeholders[1].text_frame
    
    p = tf.add_paragraph()
    p.text = "Obiectiv: Clasificarea ierarhică a culturilor agricole folosind serii de timp satelitare (Sentinel-2)."
    p = tf.add_paragraph()
    p.text = "Provocarea majoră (Class Imbalance): Setul de date este puternic dezechilibrat."
    p = tf.add_paragraph()
    p.text = "Clasele dominante (ex: pajiștile) copleșesc antrenamentul."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Clasele rare, cum ar fi livezile, reprezintă sub 0.45% din date."
    p.level = 1
    p.font.color.rgb = COLOR_ORANGE
    p = tf.add_paragraph()
    p.text = "Limitările modelelor standard (ex: MSResNet monolitic):"
    p = tf.add_paragraph()
    p.text = "Acuratețe globală aparent mare (~80.8%), dar ignoră clasele rare (recall ~16% pe livezi)."
    p.level = 1
    p.font.bold = True
    apply_text_style(tf)

    # ==========================================
    # SLIDE 3: Baza Teoretică și Lucrări de Referință
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "2. Baza Teoretică și Lucrări de Referință"
    tf = slide.placeholders[1].text_frame
    
    tf.text = "Seturi de date și Baseline:"
    p = tf.add_paragraph()
    p.text = "Rußwurm et al. (2019) – BreizhCrops: Baseline MSResNet și vulnerabilitatea long-tail."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Arhitecturi Temporale Specializate:"
    p = tf.add_paragraph()
    p.text = "Garnot et al. (2020) – LTAE & GL-TAE: Atenție temporală pentru fenologie."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Modele Ierarhice (MoE) și Few-Shot:"
    p = tf.add_paragraph()
    p.text = "Lu et al. (2024) – Uncertainty Mixture of Experts...: MoE ca soluție SOTA în agricultură."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Snell et al. (2017) – Prototypical Networks: Spații metrice pentru clase rare."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Fuziune Tabulară și Priori Fizici:"
    p = tf.add_paragraph()
    p.text = "Chen & Guestrin (2016) – XGBoost: Late-fusion cost-sensitive pe date heterogene."
    p.level = 1
    apply_text_style(tf, font_size=16)

    # ==========================================
    # SLIDE 4: Setul de Date și Protocolul
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "3. Setul de Date: BreizhCrops & Protocolul Regional"
    tf = slide.placeholders[1].text_frame
    
    p = tf.add_paragraph()
    p.text = "Sursa Datelor: Sentinel-2 L1C (Top of Atmosphere Reflectance)."
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Nivelul L1C reține norii și umbrele. Modelele neurale trebuie să învețe nativ să ignore observațiile bruiate (prin mecanisme de Atenție)."
    p.level = 1
    p.font.color.rgb = COLOR_BLUE
    p = tf.add_paragraph()
    p.text = "Structura Intrării: Tensor (Batch, 45 Timesteps, 13 Canale)."
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Protocolul de Validare (Regional Generalization):"
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Train: Regiunile frh01, frh02, frh03 (485.649 parcele)."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Eval (Held-out Test): Regiunea frh04 (122.614 parcele)."
    p.level = 1
    apply_text_style(tf)

    # ==========================================
    # SLIDE 5: Ierarhia și Dezechilibrul
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "4. Ierarhia Claselor și Dezechilibrul Extrem"
    tf = slide.placeholders[1].text_frame
    
    tf.text = "Ierarhia procesată de model (9 clase globale):"
    
    p = tf.add_paragraph()
    p.text = "1. Cereale de iarnă: Orz (0), Grâu (1)"
    p = tf.add_paragraph()
    p.text = "2. Culturi terminale: Rapiță (2), Porumb (3)"
    p = tf.add_paragraph()
    p.text = "3. Pajiști (Clasele Dominante): Perm. (7), Temp. (8)"
    p.font.color.rgb = COLOR_BLUE
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "4. Clase Minoritare: Floarea-soarelui (4), Livezi (5), Nuci (6)"
    p.font.color.rgb = COLOR_ORANGE
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Provocarea statistică a proiectului:"
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "În regiunea de test (frh04), pajiștile au peste 64.000 de parcele, în timp ce Livezile reprezintă un bottleneck de doar 553 de exemple (~0.45%)."
    p.level = 1
    apply_text_style(tf)

    # ==========================================
    # SLIDE 6: Arhitectura Vizuala (Poza 1)
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "5. Arhitectura Vizuală: Arborele DMOE & Meta-Classifier"
    
    img_path_1 = 'dmoe_diagram.png'
    if os.path.exists(img_path_1):
        slide.shapes.add_picture(img_path_1, Inches(0.5), Inches(2.0), width=Inches(9.0))
    else:
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(2.0), Inches(9.0), Inches(2.0))
        txBox.text_frame.text = f"[ EROARE: Pune fișierul '{img_path_1}' (Mermaid LR) în folder! ]"
        txBox.text_frame.paragraphs[0].font.color.rgb = COLOR_RED

    # ==========================================
    # SLIDE 7: Justificarea Selectiei
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "6. Justificarea Selecției Arhitecturale"
    tf = slide.placeholders[1].text_frame
    
    tf.text = "Specializarea modelelor pe baza caracteristicilor claselor:"
    p = tf.add_paragraph()
    p.text = "Nivelul L2c (Pajiști): Model MSResNet"
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Zeci de mii de parcele. Rețelele reziduale adânci învață limite complexe pe volume mari."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Nivelurile L2a (Cereale) & L2b: Modele LTAE / GL-TAE"
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Orzul și grâul diferă fenologic extrem de puțin. Modelele extrag strict momentele critice din serii."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Nivelul L3 (Livezi / Arbori): Model ProtoNet"
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Sub 500 de sample-uri. Calculăm distanțe matematice (metric learning) față de un „prototip ideal”."
    p.level = 1
    apply_text_style(tf)

    # ==========================================
    # SLIDE 8: DMOE Handcrafted Failure
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "7. Limitarea Fuziunii Manuale (Handcrafted DMOE)"
    tf = slide.placeholders[1].text_frame
    
    tf.text = "Abordarea inițială:"
    p = tf.add_paragraph()
    p.text = "Fuziune matematică manuală folosind praguri fixe și înmulțirea probabilităților între nivelurile L1, L2 și L3."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Problema structurală (Compounding Errors):"
    p.font.color.rgb = COLOR_RED
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Dacă L1 greșea o ramură, eroarea se propaga ireversibil în L2. Acuratețea globală a cascadelor a stagnat la ~75-78%."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Soluția (Extragerea Necondiționată):"
    p.font.color.rgb = COLOR_GREEN
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Decuplarea ierarhiei. Transformarea DMOE dintr-un arbore de decizie strict într-un „Feature Extractor” masiv."
    p.level = 1
    apply_text_style(tf)

    # ==========================================
    # SLIDE 9: Vector Diagram (Poza 2 - XGBoost.png)
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "8. Schema Conceptuală: Vectorul de Decizie (64-D)"
    
    img_path_2 = 'XGBoost.png'
    if os.path.exists(img_path_2):
        slide.shapes.add_picture(img_path_2, Inches(0.5), Inches(2.0), width=Inches(9.0))
    else:
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(2.0), Inches(9.0), Inches(2.0))
        txBox.text_frame.text = f"[ EROARE: Pune fișierul '{img_path_2}' în folder! ]"
        txBox.text_frame.paragraphs[0].font.color.rgb = COLOR_RED

    # ==========================================
    # SLIDE 10: Vector Synthesis (Text)
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "9. Anatomia Matematică a Vectorului (64-D)"
    tf = slide.placeholders[1].text_frame
    
    tf.text = "Arhitectura Late-Fusion transformă predicțiile neurale în date tabulare de înaltă precizie:"
    
    p = tf.add_paragraph()
    p.text = "1. Logiții Bruți ai Experților (15-D):"
    p.font.bold = True
    p.font.color.rgb = COLOR_BLUE
    p = tf.add_paragraph()
    p.text = "Ieșirile brute (L1, L2a, L2b, L2c, L3) păstrează distanța matematică a confidenței înainte de Softmax."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "2. Ingineria Confidenței (Marginile):"
    p.font.bold = True
    p.font.color.rgb = COLOR_BLUE
    p = tf.add_paragraph()
    p.text = "Pe lângă modelele globale, extragem margini specifice (ex: diferența de probabilitate dintre Orz și Grâu) pentru departajare forțată."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "3. Ancorele Spațio-Temporale (Metadate 6-D):"
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN
    p = tf.add_paragraph()
    p.text = "Spațiu: Amprenta (Area) și corecția climatică (Latitudine)."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Timp: Densitatea biomasei (max_ndvi) și momentul maturității (peak_doy)."
    p.level = 1
    apply_text_style(tf, font_size=16)

    # ==========================================
    # SLIDE 11: Workflow End-to-End
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "10. Workflow-ul End-to-End de Producție"
    tf = slide.placeholders[1].text_frame
    
    tf.text = "Arhitectura finală implementată pentru inferență (frh04):"
    p = tf.add_paragraph()
    p.text = "1. Pasarea imaginilor Sentinel-2 prin ramurile neurale înghețate."
    p = tf.add_paragraph()
    p.text = "2. Extragerea la cerere a features-urilor fizice și calcularea marginilor matematice."
    p = tf.add_paragraph()
    p.text = "3. Generarea tensorului final de decizie 64-D."
    p = tf.add_paragraph()
    p.text = "4. Clasificarea finală prin XGBoost (Late-Fusion)."
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "Calibrarea Cost-Sensitive: Am aplicat un multiplicator (m=25) care penalizează agresiv ratarea claselor rare (Livezi) în antrenamentul XGBoost, stabilizând recall-ul."
    p.level = 1
    p.font.color.rgb = COLOR_BLUE
    apply_text_style(tf)

    # ==========================================
    # SLIDE 12: Rezultate Finale
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "11. Rezultate Finale (frh04 Regional Protocol)"
    
    rows, cols = 5, 4
    left, top, width, height = Inches(0.5), Inches(1.8), Inches(9.0), Inches(2.0)
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    
    headers = ["Configurație", "Acuratețe Globală", "Recall Livezi", "Status Ținte"]
    for i, col_name in enumerate(headers):
        table.cell(0, i).text = col_name
        table.cell(0, i).text_frame.paragraphs[0].font.bold = True
        table.cell(0, i).text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        table.cell(0, i).fill.solid()
        table.cell(0, i).fill.fore_color.rgb = COLOR_BLUE

    data = [
        ["Ținte Proiect", "> 80.69%", "> 35.00%", "Referință"],
        ["MSResNet (Baseline)", "80.77%", "15.73%", "Eșec major (Livezi)"],
        ["XGBoost (58-d Neural, m=40)", "80.33%", "37.97%", "Eșec minor (Acc)"],
        ["XGBoost (64-d Complet, m=25)", "80.76%", "35.80%", "SUCCES INTEGRAL"]
    ]
    
    for row_idx, row_data in enumerate(data):
        for col_idx, cell_data in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = cell_data
            cell.text_frame.paragraphs[0].font.size = Pt(16)
            cell.text_frame.paragraphs[0].font.color.rgb = COLOR_TEXT
            
            if row_idx == 3:
                cell.text_frame.paragraphs[0].font.color.rgb = COLOR_GREEN
                cell.text_frame.paragraphs[0].font.bold = True
            if row_idx == 1 and col_idx == 2:
                cell.text_frame.paragraphs[0].font.color.rgb = COLOR_RED

    # ==========================================
    # SLIDE 13: Concluzii
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "12. Concluzii"
    tf = slide.placeholders[1].text_frame
    
    tf.text = "1. Meta-Classifiers rezolvă 'Paradoxul Router-ului':"
    p = tf.add_paragraph()
    p.text = "Ierarhiile stricte cedează sub propria complexitate (compounding errors). Convertirea experților în 'feature extractors' și lăsarea unui arbore de decizie (XGBoost) să analizeze marginile este o soluție superioară."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "2. Importanța Metadatelor Fizice:"
    p.text = "Datele pur spectrale au limite computaționale (ex: schimbări climatice regionale). Extragerea Ariei și a vârfului fenologic (DOY) previne confuzia modelului pe date nevăzute."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "\nVă mulțumesc pentru atenție!"
    p.font.bold = True
    p.font.color.rgb = COLOR_BLUE
    apply_text_style(tf)

    # ==========================================
    # SLIDE 14: Bibliografie Selectivă
    # ==========================================
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "13. Bibliografie Selectivă"
    tf = slide.placeholders[1].text_frame
    
    tf.text = "1. Rußwurm, M., et al. (2019). BreizhCrops: A Time Series Dataset for Crop Type Mapping. ISPRS Archives."
    p = tf.add_paragraph()
    p.text = "2. Lu, Q., et al. (2024). Uncertainty Mixture of Experts Model for Long Tail Crop Type Mapping. Remote Sensing, 16(6), 1026."
    p = tf.add_paragraph()
    p.text = "3. Garnot, V. S. F., et al. (2020). Satellite Image Time Series Classification with Pixel-Set Encoders and Temporal Self-Attention. CVPR 2020."
    p = tf.add_paragraph()
    p.text = "4. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-shot Learning. NeurIPS, 30."
    p = tf.add_paragraph()
    p.text = "5. Bannari, A., et al. (1995). A review of vegetation indices. Remote Sensing Reviews, 13(1-2), 95-120."
    p = tf.add_paragraph()
    p.text = "6. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16."
    apply_text_style(tf, font_size=14)

    prs.save('AgricultureAi_Prezentare_Master_V2.pptx')
    print("Prezentarea V2 a fost generată cu succes! Fișierul a fost salvat ca 'AgricultureAi_Prezentare_Master_V2.pptx'")

if __name__ == '__main__':
    create_presentation()