PROJECT_DESCRIPTION_PROMPT = """
    You are a project analyst tasked with creating a comprehensive project description. Based on the document content and categorization results, generate a detailed project description.

    ### Document Content:
    {context}

    ### Project Classification Results:
    - Business Category: {business_category}
    - Business Size: {business_size}
    - Service Category: {service_category}
    - Industry Category: {industry_category}
    - Service Type: {service_type}

    ### Task:
    Create a comprehensive project description that includes but not limited to the following aspects:
    1. **Main Purpose**: What the project aims to achieve
    2. **Key Features**: Primary functionalities and capabilities
    3. **Target Audience**: Who will use this solution and their characteristics
    4. **Industry Context**: How it fits within the identified industry
    5. **Business Model**: How it aligns with the business category and size
    6. **Service Approach**: How the service type influences the delivery

    ### Requirements:
    - Keep the description concise but comprehensive 
    - Ensure consistency with the selected category labels
    - Use professional, clear language
    - Focus on practical business value
    - Make it suitable for stakeholder communication

    ### Output Format:
    Provide only the project description without additional formatting or headers.
    """