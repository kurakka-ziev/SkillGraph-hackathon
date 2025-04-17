import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import tabula
import pandas as pd

#  Функции извлечения и обработки данных
def extract_table_from_pdf_tabula(pdf_path):
    try:
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        if tables:
            combined_df = pd.concat(tables, ignore_index=True)
            table_data = combined_df.values.tolist()
            return table_data
        else:
            return []
    except Exception as e:
        print(f"Ошибка при извлечении таблицы: {e}")
        return []

def create_employer_matrix(table_data):
    employer_matrix = {}
    if table_data:
        headers = [str(header).strip() for header in table_data[0]]
        for row in table_data[1:]:
            competency = str(row[0]).strip()
            employer_matrix[competency] = {}
            for i in range(1, len(headers)):
                try:
                    level = int(float(str(row[i]).strip()))
                    employer_matrix[competency][headers[i]] = level
                except (ValueError, TypeError):
                    pass
    return employer_matrix

def create_candidate_matrix(diploma_data, other_skills):
    candidate_matrix = {}
    grade_to_level = {
        "удовлетворительно": 1,
        "хорошо": 2,
        "отлично": 3
    }
    for subject, grade in diploma_data.items():
        if subject == "Статистические методы":
            candidate_matrix["Статистические методы и первичный анализ данных"] = grade_to_level[grade]
        elif subject == "Машинное обучение":
            candidate_matrix["Методы машинного обучения"] = grade_to_level[grade]
        elif subject == "Базы данных":
            candidate_matrix["SQL базы данных"] = grade_to_level[grade]
            candidate_matrix["NoSQL базы данных"] = grade_to_level[grade]
            
    candidate_matrix.update(other_skills)
    return candidate_matrix

def create_graph_data(employer_matrix, candidate_matrix, profession):
    nodes = list(employer_matrix.keys()) + list(candidate_matrix.keys())
    nodes = list(set(nodes))
    node_features = []
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    for node in nodes:
        emp_level = employer_matrix.get(node, {}).get(profession, 0)
        cand_level = candidate_matrix.get(node, 0)
        node_features.append([emp_level, cand_level])
    
    edges = []
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node1 != node2:
                if candidate_matrix.get(node1, 0) > 0 and candidate_matrix.get(node2, 0) > 0:
                    edges.append((i, j))
    
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return x, edge_index, node_to_idx

# Модель GNN (новая!!!)
class CompetencyGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_competencies):  #  добавили num_competencies
        super(CompetencyGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, num_competencies)  #  выход для каждой компетенции

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.fc(x)  #  Теперь выход - вектор
        return x

# Функция обучения 
def train_gnn(model, x, edge_index, target_levels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, target_levels)  #  сравниваем по каждой компетенции
    loss.backward()
    optimizer.step()
    return out.detach().numpy()  #  вохвращаем numpy array

# Функция анализа пробелов (!!!)
def analyze_competency_gaps(model_output, employer_matrix, node_to_idx, profession):
    gaps = {}
    for competency, idx in node_to_idx.items():
        required_level = employer_matrix.get(competency, {}).get(profession, 0)
        predicted_level = model_output[idx]

        #  Нужно определить порог для пробела (!!!)
        if required_level > predicted_level + 0.5:
            gaps[competency] = {
                "required": required_level,
                "predicted": predicted_level,
                "difference": required_level - predicted_level
            }
    return gaps

#  мок
pdf_path = "Матрица.pdf"
table_data = extract_table_from_pdf_tabula(pdf_path)
employer_matrix = create_employer_matrix(table_data)

diploma_data = {
    "Статистические методы": "хорошо",
    "Машинное обучение": "отлично",
    "Базы данных": "отлично"
}
other_skills = {
    "Python": 3,
    "C++": 2,
    "Графовые нейросети": 1,
    "Hadoop, SPARK, Hive": 2
}
candidate_matrix = create_candidate_matrix(diploma_data, other_skills)

profession_to_compare = "АНАЛИТИК ДАННЫХ\nIDATA SCIENTIST, ML\nENGINEERI"
x, edge_index, node_to_idx = create_graph_data(employer_matrix, candidate_matrix, profession_to_compare)

input_dim = 2
hidden_dim = 16
output_dim = 8
num_competencies = len(node_to_idx)  #  количество компетенций
model = CompetencyGNN(input_dim, hidden_dim, output_dim, num_competencies)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# target_levels на основе employer_matrix
target_levels = torch.zeros(num_competencies, dtype=torch.float)
for competency, idx in node_to_idx.items():
    target_levels[idx] = employer_matrix.get(competency, {}).get(profession_to_compare, 0)

# обучение
model_output = train_gnn(model, x, edge_index, target_levels, optimizer, criterion)

# анализ пробелов
competency_gaps = analyze_competency_gaps(model_output, employer_matrix, node_to_idx, profession_to_compare)

print("Пробелы в компетенциях:")
for skill, gap in competency_gaps.items():
    print(f"- {skill}: Требуется {gap['required']}, предсказано {gap['predicted']:.2f}, разница {gap['difference']:.2f}")