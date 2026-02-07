#!/usr/bin/env python3
"""
邮件发送辅助模块
===============

支持多种邮件发送方式:
1. Gmail SMTP (推荐)
2. 其他 SMTP 服务器
3. 系统 mail 命令
4. 禁用邮件功能

使用方法:
1. 修改下面的配置
2. 运行测试: python email_helper.py --test
"""

import smtplib
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

# ============================================================================
# 邮件配置 - 请根据您的需求修改
# ============================================================================

# 方式1: 使用 Gmail (推荐)
# 需要: 1) 开启 "允许安全性较低的应用" 或 2) 使用应用专用密码
# 获取应用专用密码: Google账户 -> 安全性 -> 两步验证 -> 应用专用密码
GMAIL_CONFIG = {
    "enabled": True,  # 设为 True 启用 Gmail
    "email": "byboyuanzhang@gmail.com",
    "app_password": "ipah sgzw ozpa ctcy",  # Gmail 应用专用密码
}

# 方式2: 使用其他 SMTP 服务器
SMTP_CONFIG = {
    "enabled": False,  # 设为 True 启用
    "server": "smtp.example.com",
    "port": 587,
    "use_tls": True,
    "username": "your_username",
    "password": "your_password",
    "from_email": "sender@example.com",
}

# 方式3: 使用系统 mail 命令 (Linux)
SYSTEM_MAIL_CONFIG = {
    "enabled": False,  # 设为 True 启用
}

# 收件人配置
RECIPIENT_EMAIL = "byboyuanzhang@gmail.com"

# 总开关 - 设为 False 完全禁用邮件功能
EMAIL_ENABLED = True


# ============================================================================
# 邮件发送函数
# ============================================================================

def send_email(
    subject: str, 
    body: str, 
    is_error: bool = False,
    recipient: Optional[str] = None
) -> bool:
    """
    发送邮件通知
    
    Args:
        subject: 邮件主题
        body: 邮件内容
        is_error: 是否为错误通知（高优先级）
        recipient: 收件人（默认使用配置中的收件人）
    
    Returns:
        是否发送成功
    """
    if not EMAIL_ENABLED:
        print("📧 邮件功能已禁用")
        return False
    
    recipient = recipient or RECIPIENT_EMAIL
    full_subject = f"[ABC调参] {subject}"
    
    # 尝试方式1: Gmail
    if GMAIL_CONFIG["enabled"]:
        try:
            return _send_via_gmail(full_subject, body, recipient, is_error)
        except Exception as e:
            print(f"⚠️ Gmail 发送失败: {e}")
    
    # 尝试方式2: 其他 SMTP
    if SMTP_CONFIG["enabled"]:
        try:
            return _send_via_smtp(full_subject, body, recipient, is_error)
        except Exception as e:
            print(f"⚠️ SMTP 发送失败: {e}")
    
    # 尝试方式3: 系统 mail 命令
    if SYSTEM_MAIL_CONFIG["enabled"]:
        try:
            return _send_via_system_mail(full_subject, body, recipient)
        except Exception as e:
            print(f"⚠️ 系统邮件命令失败: {e}")
    
    print("⚠️ 所有邮件发送方式均失败或未配置")
    return False


def _send_via_gmail(subject: str, body: str, recipient: str, is_error: bool) -> bool:
    """通过 Gmail SMTP 发送"""
    msg = MIMEMultipart()
    msg["From"] = GMAIL_CONFIG["email"]
    msg["To"] = recipient
    msg["Subject"] = subject
    
    if is_error:
        msg["X-Priority"] = "1"
    
    msg.attach(MIMEText(body, "plain", "utf-8"))
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(GMAIL_CONFIG["email"], GMAIL_CONFIG["app_password"])
        server.sendmail(GMAIL_CONFIG["email"], recipient, msg.as_string())
    
    print(f"📧 邮件已通过 Gmail 发送: {subject}")
    return True


def _send_via_smtp(subject: str, body: str, recipient: str, is_error: bool) -> bool:
    """通过自定义 SMTP 服务器发送"""
    msg = MIMEMultipart()
    msg["From"] = SMTP_CONFIG["from_email"]
    msg["To"] = recipient
    msg["Subject"] = subject
    
    if is_error:
        msg["X-Priority"] = "1"
    
    msg.attach(MIMEText(body, "plain", "utf-8"))
    
    if SMTP_CONFIG["use_tls"]:
        with smtplib.SMTP(SMTP_CONFIG["server"], SMTP_CONFIG["port"], timeout=30) as server:
            server.starttls()
            server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
            server.sendmail(SMTP_CONFIG["from_email"], recipient, msg.as_string())
    else:
        with smtplib.SMTP_SSL(SMTP_CONFIG["server"], SMTP_CONFIG["port"], timeout=30) as server:
            server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
            server.sendmail(SMTP_CONFIG["from_email"], recipient, msg.as_string())
    
    print(f"📧 邮件已通过 SMTP 发送: {subject}")
    return True


def _send_via_system_mail(subject: str, body: str, recipient: str) -> bool:
    """通过系统 mail 命令发送"""
    result = subprocess.run(
        ["mail", "-s", subject, recipient],
        input=body.encode(),
        timeout=30,
        capture_output=True
    )
    
    if result.returncode == 0:
        print(f"📧 邮件已通过系统命令发送: {subject}")
        return True
    else:
        raise Exception(f"mail 命令返回码: {result.returncode}")


def test_email():
    """测试邮件发送"""
    print("=" * 50)
    print("邮件发送测试")
    print("=" * 50)
    print(f"收件人: {RECIPIENT_EMAIL}")
    print(f"Gmail 启用: {GMAIL_CONFIG['enabled']}")
    print(f"SMTP 启用: {SMTP_CONFIG['enabled']}")
    print(f"系统邮件启用: {SYSTEM_MAIL_CONFIG['enabled']}")
    print("=" * 50)
    
    success = send_email(
        subject="测试邮件",
        body="这是一封测试邮件，用于验证 ABC Vector 超参数搜索的邮件通知功能是否正常工作。\n\n"
             "如果您收到这封邮件，说明配置成功！",
        is_error=False
    )
    
    if success:
        print("\n✅ 邮件发送成功！请检查收件箱。")
    else:
        print("\n❌ 邮件发送失败。请检查配置。")
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="邮件发送测试工具")
    parser.add_argument("--test", action="store_true", help="发送测试邮件")
    args = parser.parse_args()
    
    if args.test:
        test_email()
    else:
        print("使用 --test 参数发送测试邮件")
        print("例如: python email_helper.py --test")