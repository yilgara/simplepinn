import simplepinn as sp


model = sp.solve(
    equation="heat",
    alpha=0.01,
    epochs=2000,
)

model.plot(t=0.5)
