import torch
import torch.nn as nn

# Training data
# [study_hours, attendance, previous_score, sleep_hours]

X = torch.tensor([
    [2., 70., 55., 6.],
    [4., 80., 65., 7.],
    [6., 90., 75., 8.],
    [8., 95., 85., 7.],
    [3., 75., 60., 8.],
    [7., 88., 78., 6.],
    [5., 85., 70., 7.],
    [9., 98., 90., 8.]
])

# Final scores
y = torch.tensor([
    [52.],
    [65.],
    [80.],
    [91.],
    [58.],
    [83.],
    [72.],
    [95.]
])

# Model
# 4 inputs → 1 output
model = nn.Linear(4, 1)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# Training
for epoch in range(5000):

    prediction = model(X)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:
        print(epoch, loss.item())


# User input
study_hours = float(input("Study hours: "))
attendance = float(input("Attendance (%): "))
previous_score = float(input("Previous score: "))
sleep_hours = float(input("Sleep hours: "))

X_new = torch.tensor([[
    study_hours,
    attendance,
    previous_score,
    sleep_hours
]])

# Prediction
new_prediction = model(X_new)

print("Predicted final score:", new_prediction.item())