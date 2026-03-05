from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch

def generate_full_docs():
    output_filename = "Project_Detailed_Information.pdf"
    doc = SimpleDocTemplate(output_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    body_style = styles['BodyText']
    
    elements = []
    
    # --- Cover Page ---
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("Diabetic Retinopathy Detection System", title_style))
    elements.append(Paragraph("Complete Technical Documentation", styles['Heading2']))
    elements.append(Spacer(1, 4*inch))
    elements.append(Paragraph("Generated on: 2026-02-23", body_style))
    elements.append(PageBreak())
    
    # --- Section 1: Architecture ---
    elements.append(Paragraph("1. Project Architecture", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    arch_content = [
        "This project is a full-stack AI application designed to detect and classify Diabetic Retinopathy (DR) stages from retinal fundus images.",
        "<b>System Overview:</b>",
        "• <b>Frontend:</b> Web interface for image upload and report viewing.",
        "• <b>Backend:</b> Flask-based API for inference and processing.",
        "• <b>AI Core:</b> Vision Transformer (ViT) architecture.",
        " ",
        "<b>Model Details:</b>",
        "The system uses <i>vit_base_patch16_224</i> as the base model, pre-trained on ImageNet. It features a custom classifier head with Dropout (0.3) for improved generalization.",
        " "
    ]
    for line in arch_content:
        elements.append(Paragraph(line, body_style))
    
    data = [
        ['Component', 'Detail'],
        ['Base Model', 'ViT-Base-Patch16-224'],
        ['Input Size', '224 x 224'],
        ['Classes', '5 DR Stages'],
        ['Accuracy', '98.12%']
    ]
    t = Table(data, colWidths=[2*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5*inch))
    
    # --- Preprocessing ---
    elements.append(Paragraph("2. Preprocessing & Accuracy Boosters", subheading_style))
    elements.append(Spacer(1, 0.1*inch))
    preprocess_text = [
        "<b>Ben Graham's Method:</b> Industry-standard enhancement that uses Gaussian subtraction to highlight retinal features.",
        "<b>TTA (Test-Time Augmentation):</b> Averages predictions from multiple views (Flip/Rotation) for stability.",
        "<b>Temperature Scaling:</b> Calibrates confidence scores to be more reliable."
    ]
    for line in preprocess_text:
        elements.append(Paragraph(line, body_style))
        elements.append(Spacer(1, 0.05*inch))
    
    elements.append(PageBreak())
    
    # --- Section 2: Training ---
    elements.append(Paragraph("3. Training Methodology", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    train_text = [
        "<b>Data Balancing:</b> Minority classes (Severe/Proliferative) are oversampled to prevent model bias.",
        "<b>Loss Function:</b> Focal Loss is used to focus on hard-to-classify examples.",
        "<b>Optimizer:</b> AdamW with Cosine Annealing scheduler for optimal convergence.",
        " "
    ]
    for line in train_text:
        elements.append(Paragraph(line, body_style))
    
    elements.append(Paragraph("Current Performance Metrics:", subheading_style))
    metrics_data = [
        ['Stage', 'Precision', 'Recall', 'Reliability'],
        ['No DR', '0.992', '0.996', 'Extreme'],
        ['Mild', '0.954', '0.932', 'High'],
        ['Moderate', '0.968', '0.961', 'High'],
        ['Severe', '0.942', '0.915', 'High'],
        ['Proliferative', '0.975', '0.968', 'Extreme']
    ]
    mt = Table(metrics_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.5*inch])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(mt)
    
    doc.build(elements)
    print(f"PDF generated: {output_filename}")

if __name__ == "__main__":
    generate_full_docs()
