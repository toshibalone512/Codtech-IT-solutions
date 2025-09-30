import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Read Data
df = pd.read_csv("sales_data.csv")

# Analyze Data
summary = df.groupby("Product")["Sales"].sum().reset_index()

# Save plot
plt.figure(figsize=(6,4))
plt.bar(summary["Product"], summary["Sales"], color=["tomato", "orange"])
plt.title("Total Sales by Product")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.savefig("sales_chart.png")
plt.close()

# Generate PDF Report
pdf_file = "Sales_Report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=A4)
elements = []

# Styles
styles = getSampleStyleSheet()
title = Paragraph("Sales Report", styles['Title'])
elements.append(title)
elements.append(Spacer(1, 20))

# Table Data
table_data = [["Date", "Product", "Sales"]] + df.values.tolist()
table = Table(table_data)

# Table Styling
table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.grey),
    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
    ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0,0), (-1,0), 12),
    ('BACKGROUND',(0,1),(-1,-1),colors.beige),
    ('GRID', (0,0), (-1,-1), 1, colors.black),
]))
elements.append(table)
elements.append(Spacer(1, 20))

# Add Chart
elements.append(Paragraph("Sales Summary Chart", styles['Heading2']))
elements.append(Spacer(1, 12))
elements.append(Image("sales_chart.png", width=400, height=250))

# Save PDF
doc.build(elements)
print(f"Report generated successfully: {pdf_file}")
