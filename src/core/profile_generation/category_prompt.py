class ProfileCategoryPrompt:
    def __init__(self):
        # Base prompt template
        self.base_prompt = """
### Role
You are a **classification assistant**. Your task is to assign the most accurate label for the given category based on the retrieved RFP document content.  

### Instructions
You will receive:
1. **Document content (summarized from graph retrieval)** – this contains the relevant information extracted from the RFP.
2. **Category Query** – the specific classification category you need to assign.
3. **Available Labels** – the set of possible labels for the category.
4. **Category Definitions** – descriptions of each label to guide accurate classification.

Your responsibilities:
1. **Read carefully** the document content and understand its intent and scope.  
2. **Compare thoroughly** against each label definition. Pay attention to:  
   - The context (procurement, services, industry, etc.).  
   - The purpose and function described.  
   - Any explicit or implied references in the text.  
3. **Select exactly one label** that best represents the content with respect to the category query.  
4. If the content **does not provide enough evidence** to confidently choose among the labels, return: `Unknown`.  

### Output Rules
- Respond with **only** the label name (exactly as listed in the available labels).  
- Do **not** provide explanation, reasoning, or additional text.  
- The chosen label must strictly belong to the provided label set.  

### Inputs
- **Document content (summarized from graph retrieval):**  
{aggregated_content}

- **Category Query:**  
{category_query}

- **Available Labels for {category}:**  
{label_list}

- **Category Definitions and Descriptions:**  
{definitions}

### Final Answer

"""
        # Category definitions
        self.category_definitions = {
            "BusinessCategory": {
                "title": "## Project Business Category",
                "definitions": {
                    "B2B": "Business-to-Business, projects where the main customers are other companies or organizations.",
                    "B2C": "Business-to-Consumer, projects targeting individual end users or general consumers.",
                    "B2E": "Business-to-Employee, projects providing services or platforms for a company's internal staff.",
                    "B2G": "Business-to-Government, projects serving government bodies, agencies, or public sector entities.",
                    "C2C": "Consumer-to-Consumer, projects enabling transactions or interactions directly between individual consumers."
                }
            },
            
            "BusinessSize": {
                "title": "## Project Business Size",
                "definitions": {
                    "Small and Middle-sized": "Projects designed for small to medium-sized businesses with limited scale, budget, and resources.",
                    "Enterprise Size": "Projects aimed at large organizations with complex systems, higher budgets, and large-scale operations."
                }
            },
            
            "ServiceCategory": {
                "title": "## Project Service Category",
                "definitions": {
                    "Art & Design": "Creative services related to visual arts, design tools, or creative content.",
                    "Events": "Services for planning, managing, or promoting events and gatherings.",
                    "Entertainment": "Services providing leisure, media, or amusement content.",
                    "Personalization": "Services enabling customization of products, services, or user experiences.",
                    "Comics": "Platforms or tools for accessing, creating, or distributing comics.",
                    "Shopping": "Services supporting online or offline purchasing, retail, or e-commerce.",
                    "Sports": "Services connected to sports, fitness, and athletic activities.",
                    "Social": "Platforms enabling networking, communication, and social interaction.",
                    "Tools": "Utility-focused services enhancing productivity, functionality, or daily tasks.",
                    "News & Magazines": "Platforms for delivering news articles, magazines, or related media.",
                    "Business": "Services supporting business operations, management, or professional work.",
                    "Finance": "Services related to banking, investments, payments, or financial management.",
                    "Food & Drink": "Services around food delivery, recipes, dining, or beverages.",
                    "Lifestyle": "Services enhancing daily living, hobbies, and personal interests.",
                    "Libraries & Demo": "Resources or demo applications showcasing functionality or educational samples.",
                    "Medical": "Services in healthcare, diagnostics, and medical support.",
                    "Music & Audio": "Platforms or tools for music streaming, audio creation, or sound services.",
                    "Education": "Services for learning, teaching, training, or educational content.",
                    "Health & Fitness": "Services promoting physical health, workouts, and well-being.",
                    "Productivity": "Tools and platforms designed to improve work efficiency and task management.",
                    "Auto & Vehicles": "Services related to cars, mobility, and vehicle management.",
                    "Photography": "Tools or platforms for photo capturing, editing, and sharing.",
                    "House & Home": "Services for real estate, home improvement, and interior solutions.",
                    "Dating": "Platforms connecting people for dating or relationships.",
                    "Parenting": "Services supporting childcare, parenting, and family needs.",
                    "Books & Reference": "Platforms for books, references, and reading materials.",
                    "Maps & Navigation": "Services for mapping, navigation, and location-based support.",
                    "Communication": "Services enabling messaging, calls, or information exchange.",
                    "Weather": "Platforms providing weather forecasts, alerts, and climate insights.",
                    "Video Players & Editors": "Tools for watching, editing, and managing video content.",
                    "Beauty": "Services in cosmetics, skincare, and beauty enhancement.",
                    "Travel & Local": "Services supporting travel planning, tourism, and local discovery.",
                    "Game": "Platforms or services providing gaming and interactive entertainment.",
                    "Parents & Children": "Services designed for joint use or activities between parents and children."
                }
            },
            
            "IndustryCategory": {
                "title": "## Project Industry Category",
                "definitions": {
                    "Agriculture - Forestry": "Industries involving farming, crops, and forest management.",
                    "Fishery": "Industries related to fishing, aquaculture, and marine resources.",
                    "Mining - quarrying - gravel sampling": "Extraction industries for minerals, stones, or raw materials.",
                    "Construction industry": "Industries focused on building infrastructure, housing, or civil works.",
                    "Manufacturing industry": "Production of goods across industrial, consumer, and commercial sectors.",
                    "Electricity - gas - heat supply - water supply industry": "Utilities providing energy and essential services.",
                    "Information and communication industry": "Industries covering IT, telecommunications, and digital communication.",
                    "Transportation industry - postal service": "Logistics, shipping, transport, and postal services.",
                    "Wholesale and Retail": "Trading industries selling goods to businesses or consumers.",
                    "Finance industry - insurance industry": "Banking, investments, insurance, and financial services.",
                    "Real estate industry - goods rental industry": "Property management, real estate, and asset rental services.",
                    "Academic research - specialized / technical service industry": "Research, consulting, and technical expertise services.",
                    "Accommodation - food service business": "Hospitality, lodging, and dining-related services.",
                    "Lifestyle-related service industry - entertainment industry": "Personal services and recreational entertainment.",
                    "Education - learning support industry": "Schools, training centers, and learning support organizations.",
                    "Medical care": "Healthcare, welfare, and social care services.",
                    "Employment placement -worker dispatching industry": "Staffing, recruitment, and job placement industries.",
                    "Service industry": "General service-based industries not classified elsewhere.",
                    "Public affairs": "Government administration, public services, and public sector activities.",
                    "Unclassifiable industry": "Industries that cannot be categorized under existing classifications."
                }
            },
            
            "ServiceType": {
                "title": "## Service Type",
                "definitions": {
                    "Strategy Support (Non-Digital)": "Non-digital consulting to define business strategy, vision, and direction.",
                    "DX Consulting": "Digital transformation consulting to modernize business processes using technology.",
                    "MVP Development / POC": "Building minimum viable products or proofs of concept to validate ideas.",
                    "Main Software Development / New System": "Developing entirely new systems or platforms for clients.",
                    "Main Software Development / Existing System": "Enhancing, integrating, or maintaining existing systems.",
                    "Education": "Training programs or workshops to improve digital or business skills.",
                    "Knowledge Management": "Services to capture, organize, and share organizational knowledge.",
                    "Sales Acceleration": "Tools and strategies to optimize and accelerate sales processes."
                }
            }
        }

    def get_category_definitions_text(self, category: str, label_list: list = None) -> str:
        """
        Generate formatted definitions text for a specific category.
        If label_list is provided, only include definitions for those labels.
        """
        if category not in self.category_definitions:
            return ""
        
        category_data = self.category_definitions[category]
        definitions_text = f"{category_data['title']}\n"
        
        # If specific labels are provided, filter definitions
        if label_list:
            filtered_definitions = {
                label: definition 
                for label, definition in category_data['definitions'].items() 
                if label in label_list
            }
        else:
            filtered_definitions = category_data['definitions']
        
        # Format definitions
        for idx , (label, definition) in enumerate(filtered_definitions.items()):
            definitions_text += f"{idx}. {label}: {definition}\n"
        
        return definitions_text.strip()

    def generate_prompt(self, category: str, label_list: list, category_query: str, aggregated_content: str) -> str:
        """
        Generate a complete prompt with appropriate definitions for the given category.
        """
        # Get definitions for this category
        definitions = self.get_category_definitions_text(category, label_list)
        
        # Format the prompt
        prompt = self.base_prompt.format(
            category=category,
            label_list=', '.join(label_list),
            category_query=category_query,
            aggregated_content=aggregated_content,
            definitions=definitions
        )
        
        return prompt

    def get_all_available_categories(self) -> list:
        """
        Get list of all available categories that have definitions.
        """
        return list(self.category_definitions.keys())

    def validate_category_labels(self, category: str, label_list: list) -> dict:
        """
        Validate that all labels in label_list exist in the category definitions.
        Returns dict with validation results.
        """
        if category not in self.category_definitions:
            return {
                "valid": False,
                "error": f"Category '{category}' not found in definitions",
                "missing_labels": label_list,
                "available_labels": []
            }
        
        available_labels = list(self.category_definitions[category]['definitions'].keys())
        missing_labels = [label for label in label_list if label not in available_labels]
        
        return {
            "valid": len(missing_labels) == 0,
            "error": f"Labels not found in definitions: {missing_labels}" if missing_labels else None,
            "missing_labels": missing_labels,
            "available_labels": available_labels
        }

    def get_label_definition(self, category: str, label: str) -> str:
        """
        Get definition for a specific label in a category.
        """
        if category not in self.category_definitions:
            return f"Category '{category}' not found"
        
        if label not in self.category_definitions[category]['definitions']:
            return f"Label '{label}' not found in category '{category}'"
        
        return self.category_definitions[category]['definitions'][label]

    def update_category_definition(self, category: str, label: str, definition: str):
        """
        Add or update a definition for a specific label in a category.
        """
        if category not in self.category_definitions:
            self.category_definitions[category] = {
                "title": f"## {category}",
                "definitions": {}
            }
        
        self.category_definitions[category]['definitions'][label] = definition

    def export_definitions_to_dict(self) -> dict:
        """
        Export all definitions as a dictionary for backup or transfer.
        """
        return self.category_definitions.copy()

    def import_definitions_from_dict(self, definitions_dict: dict):
        """
        Import definitions from a dictionary.
        """
        self.category_definitions = definitions_dict.copy()