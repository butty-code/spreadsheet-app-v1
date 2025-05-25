import streamlit as st
import base64
from datetime import datetime
import json
import os

def add_donation_button():
    """
    Add a donation button section to help with API key costs.
    """
    st.markdown("---")
    st.markdown("## 💖 Support This Project")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display a QR code or icon
        try:
            st.image("generated-icon.png", width=150)
        except:
            # Fallback if image isn't available
            st.markdown("""
            <div style="width: 150px; height: 150px; background-color: #f0f2f6; 
                       border-radius: 10px; display: flex; align-items: center; 
                       justify-content: center; font-size: 24px; color: #4b92db;">
                📊
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        **Help support the ongoing development and API costs of this tool!**
        
        This app uses OpenAI's GPT-4o API for AI-powered analysis, which incurs costs with each analysis.
        Your donation helps keep this feature available and supports future improvements.
        
        **Benefits of donating:**
        - Keep the AI analysis features running
        - Support new feature development
        - Get priority support for issues
        """)
    
    # Create donation option with direct PayPal button
    st.markdown("#### Support Options")
    
    st.markdown("""
    Your support helps cover API costs, server expenses, and fuels new features.
    Click the button below to donate any amount you choose using PayPal.
    """)
    
    # PayPal hosted button
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <a href="https://www.paypal.com/donate/?hosted_button_id=EPNJQECD6GRGE" target="_blank">
            <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif" alt="Donate with PayPal" style="margin: 10px auto;">
        </a>
        <p style="font-size: 14px; margin-top: 10px;">Click to donate any amount</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Using session state for compatibility with the rest of the code
    if 'donation_amount' not in st.session_state:
        st.session_state.donation_amount = None
    
    # The donation is handled directly by the PayPal button above
    # We don't need custom amounts or additional donation interfaces
    
    # Add share links to LinkedIn
    st.markdown("---")
    st.markdown("### 🔄 Share on LinkedIn")
    st.markdown("""
    Help spread the word by sharing this app on LinkedIn! You'll help others discover this tool.
    """)
    
    share_text = "Check out this amazing Excel Analyzer with AI-powered insights and Power BI integration! It's helping me analyze complex horse racing data (and other Excel files) easily."
    app_url = "https://excel-analyzer-app.streamlit.app"  # Use your actual deployed URL when available
    linkedin_share_url = f"https://www.linkedin.com/sharing/share-offsite/?url={app_url}&title=Excel%20Analyzer%20with%20AI&summary={share_text}"
    
    st.markdown(f'''
    <a href="{linkedin_share_url}" target="_blank">
        <button style="background-color: #0077b5; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
            Share on LinkedIn
        </button>
    </a>
    ''', unsafe_allow_html=True)
    
    # Add the donation information in footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #888; font-size: 12px;">
    © {datetime.now().year} Excel Analyzer | Your donations help keep AI features running | Privacy Policy
    </div>
    """, unsafe_allow_html=True)


def feature_suggestions():
    """
    Create a section for users to submit feature suggestions.
    """
    st.markdown("---")
    st.markdown("## 💡 Suggest New Features")
    
    st.markdown("""
    We're constantly improving Excel Analyzer based on user feedback! 
    Have an idea for a new feature or improvement? Let us know below.
    
    All suggestions will be sent to our developer for review and consideration.
    """)
    
    # Initialize suggestions file if it doesn't exist
    suggestions_file = "user_suggestions.json"
    if not os.path.exists(suggestions_file):
        with open(suggestions_file, "w") as f:
            json.dump([], f)
    
    # Create form for feature suggestions
    with st.form("suggestion_form"):
        feature_idea = st.text_area("Your suggestion:", 
                                     placeholder="Describe your feature idea here...",
                                     help="What would make this app more useful for you?")
        
        email = st.text_input("Email (optional):", 
                              placeholder="your.email@example.com",
                              help="We'll only use this to follow up on your suggestion.")
        
        category = st.selectbox("Category:", 
                               ["Data Analysis", "Visualization", "AI Features", 
                                "User Interface", "Performance", "Other"],
                               help="What area of the app does your suggestion relate to?")
        
        priority = st.slider("How important is this feature to you?", 
                            min_value=1, max_value=5, value=3,
                            help="1 = Nice to have, 5 = Critical need")
        
        submit_button = st.form_submit_button("Submit Suggestion")
        
        if submit_button and feature_idea:
            # Load existing suggestions
            with open(suggestions_file, "r") as f:
                suggestions = json.load(f)
            
            # Add new suggestion
            # Include the admin email so we know where to send these suggestions
            suggestion_data = {
                "suggestion": feature_idea,
                "user_email": email,
                "category": category,
                "priority": priority,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "New",
                "admin_email": "ahmed.saidi21@gmail.com"  # Hidden from UI but stored for processing
            }
            
            suggestions.append(suggestion_data)
            
            # Save updated suggestions
            with open(suggestions_file, "w") as f:
                json.dump(suggestions, f, indent=2)
            
            st.success("Thank you for your suggestion! We'll review it and consider it for future updates.")
            
            # Display additional information about suggestion handling
            st.info("""
            **What happens next?**
            
            Your suggestion has been recorded and will be sent to our developer for review.
            We evaluate all suggestions based on feasibility and how they could benefit all users.
            """)
            
    # Add note about the development team
    st.markdown("""
    ---
    
    **Note:** All suggestions are reviewed by our development team. While we can't 
    implement every suggestion, we carefully consider each one to make Excel Analyzer 
    better for everyone.
    """)


def get_donation_html(amount):
    """
    Generate the HTML for the donation modal with PayPal or similar link.
    In a real app, this would be connected to a payment processor.
    
    Args:
        amount: Donation amount
        
    Returns:
        HTML string for donation link/modal
    """
    return f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3>Thank You for Your Support!</h3>
        <p>You've selected to donate <b>${amount}</b> to support Excel Analyzer.</p>
        <p>To complete your donation, please click the button below:</p>
        
        <a href="https://www.paypal.com/donate/?hosted_button_id=EPNJQECD6GRGE" target="_blank">
            <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif" alt="Donate with PayPal" style="margin-top: 10px; margin-bottom: 10px;">
        </a>
        
        <p style="margin-top: 15px; font-size: 12px;">
            Your donation helps maintain server costs and supports further development.
            <br>You will be redirected to PayPal to complete your donation securely.
        </p>
    </div>
    """