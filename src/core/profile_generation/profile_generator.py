import logging
import json
import os

from src.core.llm_client import LLMClientFactory
from src.core.profile_generation.description_generator_prompt import PROJECT_DESCRIPTION_PROMPT
from src.core.profile_generation.description_retriever import DescriptionContextRetriever
from src.core.profile_generation.graph_retriever import GraphRetriever
from src.core.profile_generation.category_prompt import ProfileCategoryPrompt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_CATEGORIES = [
            ("BusinessCategory", ["B2B", "B2C", "B2E", "B2G", "C2C"]),
            ("BusinessSize", ["Small and Middle-sized", "Enterprise Size"]),
            (
                "ServiceCategory",
                [
                    "Art & Design",
                    "Events",
                    "Entertainment",
                    "Personalization",
                    "Comics",
                    "Shopping",
                    "Sports",
                    "Social",
                    "Tools",
                    "News & Magazines",
                    "Business",
                    "Finance",
                    "Food & Drink",
                    "Lifestyle",
                    "Libraries & Demo",
                    "Medical",
                    "Music & Audio",
                    "Education",
                    "Health & Fitness",
                    "Productivity",
                    "Auto & Vehicles",
                    "Photography",
                    "House & Home",
                    "Dating",
                    "Parenting",
                    "Books & Reference",
                    "Maps & Navigation",
                    "Communication",
                    "Weather",
                    "Video Players & Editors",
                    "Beauty",
                    "Travel & Local",
                    "Game",
                    "Parents & Children",
                ],
            ),
            (
                "IndustryCategory",
                [
                    "Agriculture - Forestry",
                    "Fishery",
                    "Mining - quarrying - gravel sampling",
                    "Construction industry",
                    "Manufacturing industry",
                    "Electricity - gas - heat supply - water supply industry",
                    "Information and communication industry",
                    "Transportation industry - postal service",
                    "Wholesale and Retail",
                    "Finance industry - insurance industry",
                    "Real estate industry - goods rental industry",
                    "Academic research - specialized / technical service industry",
                    "Accommodation - food service business",
                    "Lifestyle-related service industry - entertainment industry",
                    "Education - learning support industry",
                    "Medical care",
                    "Employment placement -worker dispatching industry",
                    "Service industry",
                    "Public affairs",
                    "Unclassifiable industry",
                ],
            ),
            (
                "ServiceType",
                [
                    "Strategy Support (Non-Digital)",
                    "DX Consulting",
                    "MVP Development / POC",
                    "Main Software Development / New System",
                    "Main Software Development / Existing System",
                    "Education",
                    "Knowledge Management",
                    "Sales Acceleration",
                ],
            ),
        ]

class ProfileGenerator:
    def __init__(self, model: str = "gpt-4.1"):
        self.llm_factory = LLMClientFactory()
        self.prompt_generator = ProfileCategoryPrompt()
        self.retriever = GraphRetriever()
        self.description_retriever = DescriptionContextRetriever(max_context_tokens=800000)
        self.model = model

        self.project_categories = PROJECT_CATEGORIES

    def _generate_rag_query(self, category: str) -> str:
        rag_query = {
            "BusinessCategory": "Identify who the project mainly serves. Is it for other companies (B2B), individual consumers (B2C), employees inside a company (B2E), government bodies (B2G), or consumer-to-consumer use (C2C)?",
            "BusinessSize": "Determine the size of the target business. Is the solution aimed at small/medium businesses with limited resources, or large enterprises with complex operations and higher budgets?",
            "ServiceCategory": "Find the main type of service the project is about. For example finance, education, healthcare, shopping/e-commerce, entertainment, social networking, productivity tools, or gaming.",
            "IndustryCategory": "Extract the industry where the project is applied. Keep it simple, like healthcare, finance, education, IT, construction, manufacturing, or public sector.",
            "ServiceType": "Identify the kind of service needed. For example consulting, digital transformation, proof-of-concept/MVP, building a new system, upgrading an existing one, training/education, knowledge management, or sales acceleration.",
        }

        return rag_query.get(category, f"What is the {category} of the RFP document?")

    def query_category_labels(self, document_id: str, top_k: int = 5) -> dict:
        """
        For each category, use GraphRetriever to retrieve relevant context.
        Aggregate their content and use LLM to select the best fitting label from the predefined list.
        """
        selected_labels = {}

        for category, label_list in self.project_categories:
            rag_query = self._generate_rag_query(category)

            try:
                # Sử dụng GraphRetriever để lấy context liên quan đến category
                context = self.retriever.retrieve(document_id, rag_query, top_k=top_k)
                aggregated_content = context.strip()

                if aggregated_content:
                    prompt = self.prompt_generator.generate_prompt(
                        category, label_list, rag_query, aggregated_content
                    )

                    messages = [{"role": "user", "content": prompt}]
                    
                    response = self.llm_factory.chat_completion(self.model, messages)
                    if self.model.startswith("gpt"):
                        response_text = response.choices[0].message.content.strip()
                    else:
                        response_text = response.content[0].text.strip()

                    selected_label = None
                    for label in label_list:
                        if (
                            label.lower() in response_text.lower()
                            or response_text.lower() in label.lower()
                            or response_text.lower() == "unknown"
                        ):
                            selected_label = label
                            break
                    if not selected_label:
                        selected_label = "Unknown"
                        logger.warning(
                            f"Could not match LLM response '{response_text}' to any label for {category}, using default: {selected_label}"
                        )

                    selected_labels[category] = {
                        "label": selected_label,
                        "rag_query": rag_query,
                        "aggregated_text": aggregated_content,
                    }
                    logger.info(
                        f"Selected label for {category}: {selected_label} (context length: {len(aggregated_content)})"
                    )
                else:
                    selected_labels[category] = {
                        "label": "Unknown",
                        "aggregated_text": "",
                    }
                    logger.warning(
                        f"No content found for {category}, using default label: {selected_labels[category]['label']}"
                    )

            except Exception as e:
                logger.error(f"Error processing category {category}: {e}")
                selected_labels[category] = {"label": "Unknown", "aggregated_text": ""}

        return selected_labels

    def generate_project_description(self, document_id: str, selected_labels: dict) -> str:
        """
        Generate a comprehensive project description based on selected labels and retrieved content.

        Args:
            document_id: The document identifier
            selected_labels: Dictionary containing selected labels for each category

        Returns:
            A comprehensive project description string
        """
        try:
            context_result = self.description_retriever.retrieve_document_context(document_id)
            context = context_result["context"]

            logger.info(f"Retrieved context: {context_result['selected_chunks']}/{context_result['total_chunks']} chunks, "
                       f"{context_result['total_tokens']} tokens, summaries used: {context_result['used_summaries']}")

            # Extract labels for context
            business_category = selected_labels.get("BusinessCategory", {}).get("label", "Unknown")
            business_size = selected_labels.get("BusinessSize", {}).get("label", "Unknown")
            service_category = selected_labels.get("ServiceCategory", {}).get("label", "Unknown")
            industry_category = selected_labels.get("IndustryCategory", {}).get("label", "Unknown")
            service_type = selected_labels.get("ServiceType", {}).get("label", "Unknown")

            # Generate prompt for project description
            description_prompt = PROJECT_DESCRIPTION_PROMPT.format(
                context=context,
                business_category=business_category,
                business_size=business_size,
                service_category=service_category,
                industry_category=industry_category,
                service_type=service_type
            )

            # Generate description using LLM
            messages = [{"role": "user", "content": description_prompt}]
            response = self.llm_factory.chat_completion("gpt-4.1", messages)
            project_description = response.choices[0].message.content.strip()

            logger.info(f"Generated project description for {document_id} (length: {len(project_description)})")
            return project_description

        except Exception as e:
            logger.error(f"Error generating project description for {document_id}: {e}")
            return "Unable to generate project description due to processing error."

    def run(self, document_id: str, top_k: int = 5) -> dict:
        logger.info("Starting ProfileGenerator with GraphRetriever.")
        labels = self.query_category_labels(document_id, top_k=top_k)
        project_description = self.generate_project_description(document_id, labels)
        result = {
            "categories": labels,
            "project_description": project_description
        }
        return result


if __name__ == "__main__":
    pipeline = ProfileGenerator()

    DOCUMENT_ID = "4b24198d-5761-4315-a6d6-d77a80c65c31"

    logger.info("Starting RAG-based category labeling pipeline...")
    result = pipeline.run(DOCUMENT_ID, top_k=10)

    # Extract categories and description from result
    selected_labels = result["categories"]
    project_description = result["project_description"]

    # Print categories
    if selected_labels:
        logger.info("\n=== SELECTED LABELS ===")
        for category, info in selected_labels.items():
            logger.info(f"{category}: {info['label']}")
    else:
        logger.info("No labels were selected. Check if the document exists and has chunks.")

    # Print project description
    logger.info("\n=== PROJECT DESCRIPTION ===")
    logger.info(project_description)

    # Save both labels and description to file for debugging
    if not os.path.exists("new_results"):
        os.makedirs("new_results")

    output_path = os.path.join("new_results", f"{DOCUMENT_ID}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved results to {output_path}")
