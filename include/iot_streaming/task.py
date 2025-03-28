import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
def send_email_alert(result):
    sender_email = "your_email@example.com"
    receiver_email = "receiver@example.com"
    password = "your_password"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Cảnh báo chất lượng nước"

    body = f"Chỉ số ô nhiễm nước hiện tại là {result}. Mức độ ô nhiễm cao."
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
    except Exception as e:
        print(f"Không thể gửi email: {e}")