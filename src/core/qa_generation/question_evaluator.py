#!/usr/bin/env python3

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import time
import json

from src.core.qa_generation.graph_retriever import GraphRetriever
from src.core.llm_client import LLMClientFactory
from src.core.qa_generation.prompts import ANSWER_GENERATION_PROMPT

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

class QuestionEvaluator:
    def __init__(self, document_name: Optional[str] = None,
                 model_name: str = "gemini-2.5-flash",
                 graph_retriever: Optional['GraphRetriever'] = None):
        self.document_name = document_name
        self.model_name = model_name

        if graph_retriever is not None:
            self.retriever = graph_retriever
            if self.document_name is None and graph_retriever.document_name is not None:
                self.document_name = graph_retriever.document_name
        else:
            self.retriever = GraphRetriever(document_name=document_name)

        self.llm_factory = LLMClientFactory()
        self.logger = logger

        scope_info = self.document_name if self.document_name else "global_graph"
        self.logger.info(f"QuestionEvaluator initialized with scope: {scope_info}")

    def generate_answer(self, question: str, context: str, model: str = "gemini-2.5-flash") -> Dict[str, Any]:
        if not context.strip():
            return {"answer": "", "answerable": False, "related_context": "", "confidence": 0.0}

        try:
            prompt = ANSWER_GENERATION_PROMPT.format(question=question, context=context)
            response = self.retriever.call_with_retry(
                self.llm_factory.chat_completion,
                model_key=model,
                messages=prompt
            )
            raw_response = response.text if hasattr(response, "text") else str(response)

            answer, answerable, related_context = self._parse_structured_response(raw_response)
            confidence = min(0.9, len(context)/1000*0.5 + len(answer)/500*0.4) if answerable else 0.0

            return {
                "answer": answer,
                "answerable": answerable,
                "related_context": related_context,
                "confidence": confidence,
                "context_used": context,
                "model_used": model,
                "metadata": {
                    "prompt_used": prompt,
                    "raw_response": raw_response
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return {"answer": f"Error generating answer: {str(e)}", "answerable": False, "related_context": ""}

    def _parse_structured_response(self, raw_response: str):
        try:
            parts = raw_response.split("========")
            if len(parts) >= 3:
                answer = parts[0].replace("Answer:", "").strip()
                answerable = parts[1].replace("Answerable:", "").strip().lower() in ["true", "yes", "có", "đúng"]
                related_context = parts[2].replace("Related Context:", "").strip()
                return answer, answerable, related_context
            return raw_response, bool(raw_response.strip()), ""
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return raw_response, False, ""

    def ask_question(self, question: str, question_embedding: list,
                     model_name: str = "gemini-2.5-flash", limit: int = 10, use_hybrid: bool = True):
        start_time = time.time()
        try:
            if not question_embedding:
                raise ValueError("question_embedding is required and cannot be empty")

            if use_hybrid:
                search_results = self.retriever.search_knowledge_graph_hybrid(question_embedding, limit)
            else:
                search_results = self.retriever.search_knowledge_graph(question_embedding, limit)

            context = self.retriever.build_context(search_results)
            answer_result = self.generate_answer(question, context, model_name)

            return {
                "question": question,
                "answer": answer_result["answer"],
                "answerable": answer_result["answerable"],
                "related_context": answer_result["related_context"],
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return {"question": question, "answer": f"Error: {str(e)}", "error": str(e)}

    def get_questions_from_json(self):
        """
        Load questions with embeddings from JSON file
        
        Returns:
            List of question dictionaries with embeddings
        """
        import json
        
        json_file_path = "src/core/qa_generation/questions/questions.json"
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = data["questions"]
            
            self.logger.info(f"Loaded {len(questions)} questions from {json_file_path}")
            return questions
            
        except Exception as e:
            self.logger.error(f"Error reading questions JSON file: {e}")
            return []

    def check_and_create_embeddings(self, questions_data: list):
        """
        Check and create embeddings for questions based on detected language
        
        Args:
            questions_data: List of question dictionaries
            
        Returns:
            Updated questions_data with embeddings and flag indicating if file was updated
        """
        import json
        
        # Detect document language
        detected_language = self.retriever.detect_data_language(sample_size=20)
        self.logger.info(f"Detected document language: {detected_language}")
        
        # Create embedding field name based on detected language
        embedding_field = f"{detected_language.lower()}_embedding"
        
        updated = False
        json_file_path = "src/core/qa_generation/questions/questions.json"
        
        for i, question_data in enumerate(questions_data):
            # Check if embedding already exists for this language
            if embedding_field not in question_data or not question_data[embedding_field]:
                try:
                    # Translate question to detected language
                    question_translated = self.retriever.translate_question(question_data["question"])
                    
                    # Create embedding for translated question
                    embedding = self.retriever.create_embedding(question_translated)
                    
                    if embedding:
                        # Add embedding to question data
                        question_data[embedding_field] = embedding
                        updated = True
                        
                        self.logger.info(f"Created {embedding_field} for question {i+1}")
                    else:
                        self.logger.warning(f"Failed to create embedding for question {i+1}")
                        question_data[embedding_field] = []
                        
                except Exception as e:
                    self.logger.error(f"Error creating embedding for question {i+1}: {e}")
                    question_data[embedding_field] = []
            else:
                self.logger.debug(f"Question {i+1} already has {embedding_field}")
        
        # Save updated data back to JSON file if any embeddings were created
        if updated:
            try:
                # Prepare data structure for saving
                save_data = {
                    "metadata": {
                        "total_questions": len(questions_data),
                        "detected_language": detected_language,
                        "embedding_field": embedding_field,
                        "last_updated": datetime.now().isoformat()
                    },
                    "questions": questions_data
                }
                
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"Updated questions file with new {embedding_field} embeddings")
                
            except Exception as e:
                self.logger.error(f"Error saving updated questions file: {e}")
        
        return questions_data, detected_language, embedding_field

    def ask_all_questions(self, output_path: str, model_name: str = "gemini-2.5-flash"):
        """
        Answer all questions from JSON file using pre-computed embeddings.
        Ensures all questions have embeddings before processing.
        
        Args:
            output_path: Path to save JSON file with all answers
            model_name: LLM model to use for answer generation
            
        Returns:
            List of dictionaries with questions and answers
        """
        import json
        import os
        
        # Load questions from JSON file
        questions_data = self.get_questions_from_json()
        
        if not questions_data:
            self.logger.error("No questions found in JSON file")
            return []
        
        # Check and create embeddings if needed - this ensures ALL questions have embeddings
        questions_data, detected_language, embedding_field = self.check_and_create_embeddings(questions_data)
        
        # Verify that all questions now have embeddings
        questions_without_embeddings = []
        for i, question_data in enumerate(questions_data):
            if not question_data.get(embedding_field):
                questions_without_embeddings.append(i + 1)
        
        if questions_without_embeddings:
            error_msg = f"Failed to create embeddings for questions: {questions_without_embeddings}"
            self.logger.error(error_msg)
            return []
        
        # Answer all questions
        all_answers = []
        total_questions = len(questions_data)
        
        self.logger.info(f"Starting to answer {total_questions} questions using {embedding_field}...")
        self.logger.info(f"All questions have required embeddings: {embedding_field}")
        
        for i, question_data in enumerate(questions_data, 1):
            try:
                question = question_data["question"]
                question_embedding = question_data[embedding_field]  # This is guaranteed to exist now
                
                # Ask question using pre-computed embedding (required)
                result = self.ask_question(
                    question, 
                    question_embedding=question_embedding,
                    model_name=model_name
                )
                
                # Add question metadata to result
                result["question_index"] = i
                result["embedding_field_used"] = embedding_field
                
                all_answers.append(result)
                
                if i % 5 == 0 or i == total_questions:
                    self.logger.info(f"Answered {i}/{total_questions} questions")
                    
            except Exception as e:
                self.logger.error(f"Error answering question {i}: {e}")
                error_result = {
                    "question": question_data.get("question", "Unknown question"),
                    "answer": f"Error processing question: {str(e)}",
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "question_index": i,
                    "embedding_field_used": embedding_field
                }
                all_answers.append(error_result)
        
        # Save all answers to JSON
        try:
            # If output_path is just a filename, save in current directory
            if os.path.dirname(output_path) == '':
                output_path = os.path.join(os.getcwd(), output_path)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            final_results = {
                "metadata": {
                    "total_questions": total_questions,
                    "successful_answers": sum(1 for a in all_answers if a.get("success", False)),
                    "failed_answers": sum(1 for a in all_answers if not a.get("success", False)),
                    "answerable_questions": sum(1 for a in all_answers if a.get("answerable", False)),
                    "unanswerable_questions": sum(1 for a in all_answers if a.get("success", False) and not a.get("answerable", False)),
                    "all_questions_have_embeddings": True,
                    "model_used": model_name,
                    "detected_language": detected_language,
                    "embedding_field": embedding_field,
                    "timestamp": datetime.now().isoformat(),
                    "document_scope": self.document_name or "global_graph"
                },
                "answers": all_answers
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved all answers to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving answers to file: {e}")
        
        return all_answers

    def close_connection(self):
        """Close Neo4j connection"""
        try:
            if self.retriever:
                self.retriever.close_connection()
            self.logger.info("QuestionEvaluator closed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def close(self):
        """Clean up resources"""
        self.close_connection()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        self.close()


if __name__ == "__main__":
    document_name = "tmpmhh22_ge"
    evaluator = QuestionEvaluator(document_name=document_name)

    # test_question = "Có những thách thức gì?"
    # embedding = evaluator.retriever.create_embedding(test_question)

    # result = evaluator.ask_question(
    #     question=test_question,
    #     question_embedding=embedding,
    #     model_name="gemini-2.5-flash"
    # )

    # print("=== Kết quả ===")
    # print(json.dumps(result, ensure_ascii=False, indent=2))

    evaluator.ask_all_questions(f"data/output/answers/{document_name}.json")
    evaluator.close()