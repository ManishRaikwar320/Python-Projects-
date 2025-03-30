"""
Web Scraper for Real-time Data (Python + Scrapy + BeautifulSoup)
👉 यह Python Web Scraper किसी भी वेबसाइट से डेटा स्क्रैप करने के लिए बनाया गया है। इस प्रोजेक्ट में हम Amazon/Flipkart पर Product Prices ट्रैक करेंगे और Email/Telegram के जरिए Notification भेज सकते हैं।

🔹 Step-by-Step Implementation
✅ Step 1: Install Dependencies
✅ Step 2: Website से Data Extract करें
✅ Step 3: CSV में Save करें
✅ Step 4: Automate करके Price Drop Alert भेजें

""" 


import requests
from bs4 import BeautifulSoup
import pandas as pd
import smtplib
import schedule
import time
from datetime import datetime

# Amazon Product URL (अपना Product URL डालें)
URL = "https://www.amazon.in/dp/B0BSF4KY4Z"  # Example Product URL

# Headers (Amazon Bot Detection से बचने के लिए)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US, en;q=0.5"
}

# Price Scraping Function
def get_price():
    response = requests.get(URL, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")

    # Product Title Scraping
    title = soup.find("span", id="productTitle").get_text(strip=True)

    # Price Scraping
    price = soup.find("span", class_="a-price-whole").get_text(strip=True)

    print(f"Product: {title}")
    print(f"Current Price: ₹{price}")

    return title, price

# Save Data to CSV
def save_to_csv(title, price):
    filename = "price_tracker.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a", newline="") as file:
        writer = pd.DataFrame([[now, title, price]])
        writer.to_csv(filename, mode="a", header=False, index=False)
        print("✅ Data saved to CSV")

# Send Email Alert
def send_email(title, price):
    sender_email = "your_email@gmail.com"  # अपना Email डालें
    receiver_email = "receiver_email@gmail.com"  # जिसको Notification भेजना है
    password = "your_email_password"  # App Password (2-Step Authentication On करें)

    subject = f"Price Drop Alert: {title}"
    body = f"The price of {title} has dropped to ₹{price}!\nCheck here: {URL}"

    message = f"Subject: {subject}\n\n{body}"

    # Gmail SMTP Server
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

    print("📧 Email Sent!")

# Automate Scraper
def job():
    title, price = get_price()
    save_to_csv(title, price)
    if int(price.replace(",", "")) < 70000:
        send_email(title, price)

# Run Every 6 Hours
schedule.every(6).hours.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
