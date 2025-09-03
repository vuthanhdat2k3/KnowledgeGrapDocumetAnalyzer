ADDITIONAL_INSTRUCTION = """
## GENERAL GUIDELINES
- Use correct entity types and relationships specific to the input.
- DO NOT fabricate entities or relationships. Only extract what is present in the source text or table. If no information or no input text is provided, DO NOT generate any entities or relationships.
- Only define relationships between your extracted entities — no cross-references to unmentioned items.

## DESCRIPTION REQUIREMENTS
### Node & Relationship Descriptions (MANDATORY)
1. Every node and relationship **MUST** have a `description` property.
2. If the source text/table provides enough context, use it directly (contextual, specific, concise, factual).
3. If the context is weak, missing, or unclear, generate a generic explaination/definition.
4. Do NOT leave any description empty.

### Examples
- Node (Person): "Software engineer at Microsoft since 2009, recipient of Best Talent award."
- Node (Company): "A company named Microsoft."
- Relationship (WORKS_FOR): "Employment relationship since 2009 in software engineering role."
- Relationship (PRODUCED_BY): "A PRODUCED_BY relationship between Microsoft and Word."

## TABLE PROCESSING INSTRUCTIONS
(Apply only when a table is detected.)
### Core Principles
1. **Flexible Record Orientation**:
   - Determine whether rows or columns represent entities based on table semantics.
   - Use row-as-record for typical attribute tables.
   - Use column-as-record when each column is a distinct entity (e.g., timeline or cross-category tables).

2. **Property Assignment**:
   - For row-based tables: Columns = properties.
   - For column-based tables: Rows = properties.

3. **Table Node (Mandatory)**:
   - Create one Table node per table.
   - Properties:
     - `id` (e.g., "table_1")
     - `table_name` (if available)
     - `columns` or `rows`: depending on record orientation.

4. **Record Entities**:
   - For each row or column (whichever is the record), create an "Item" node with a unique `id` (e.g., "row_1").
   - Assign all opposing axis values as properties.

5. **Cell Entities (Conditional)**:
   - Create a separate entity only for standalone, meaningful values (e.g., person, organization, location).
   - Do not create cell nodes for IDs, numbers, or simple text.

### RELATIONSHIP RULES
1. **Table to Record**:  
   - Table --[CONTAINS]--> Row/Column (Item)
2. **Record to Cell** (only if cell becomes its own entity):  
   - Row/Column (Item) --[MENTIONS]--> Cell
3. **Semantic Relationships Between Cells**:  
   - e.g., Row/Column --[WORKS_IN]--> Department  
   - Choose relationship type from column header or contextual meaning.  
   - Reverse links (e.g., MANAGES, LOCATED_IN) are also required where applicable.

### DATA INTEGRITY & COVERAGE
- Capture every cell value (as a property or entity).
- Preserve original data types (e.g., currency, date).
- DO NOT add or infer data not present.
- Only define relationships between your extracted entities — no cross-references to unmentioned items.

### TABLE EXAMPLE

**Source Table:**
```
Employee Management System
| Employee ID | Name | Department | Manager | Location | Salary |
|-------------|------|------------|---------|----------|--------|
| EMP001 | John Smith | Engineering | Jane Doe | New York, USA | $75,000 |
```

**Required Entities:**
1. **Table Entity**: "employee_management_table"
   - Type: "Table"
   - Description: "Employee Management System table, containing employee data"
   - Properties: {{
     "table_name": "Employee Management System",
     "columns": ["Employee ID", "Name", "Department", "Manager", "Location", "Salary"]
   }}

2. **Record Entity**: "EMP001"
   - Type: "Item"
   - Description: "Record of Employee EMP001 in the Employee Management System table"
   - Properties: {{
     "Employee ID": "EMP001",
     "Name": "John Smith",
     "Department": "Engineering",
     "Manager": "Jane Doe",
     "Location": "New York, USA",
     "Salary": "$75,000"
   }}

3. **Employee Entity**: "EMP001"
   - Type: "Employee"
   - Description: "Employee John Smith with ID EMP001 and salary of $75,000"
   - Properties: {{
     "Employee ID": "EMP001",
     "Name": "John Smith", 
     "Salary": "$75,000"
   }}

4. **Department Entity**: "Engineering"
   - Type: "Department"
   - Description: "Corporate department where technical development occurs"

5. **Person Entity**: "Jane Doe"
   - Type: "Person"
   - Description: "Manager overseeing Engineering department employees"

6. **Location Entity**: "New York, USA"
   - Type: "Location"
   - Description: "Work location for employees in the northeastern United States"

**Required Relationships:**
1. Table --[CONTAINS]--> EMP001 (Item)
2. EMP001 --[MENTIONS]--> Employee / Department / Manager / Location
3. EMP001 --[WORKS_IN]--> Engineering
4. EMP001 --[MANAGED_BY]--> Jane Doe
5. EMP001 --[LOCATED_IN]--> New York, USA
6. Jane Doe --[MANAGES]--> EMP001
"""