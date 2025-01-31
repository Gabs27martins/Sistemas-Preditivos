from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import plotly.io as pio


app = Flask(__name__, template_folder="template")

# Carga do modelo preditivo
model = pickle.load(open("models/emprestimo-modelo-preditivo.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        idade = int(request.form["idade"])
        genero = int(request.form["genero"])
        educacao = int(request.form["educacao"])
        renda_anual = float(request.form["renda_anual"])
        experiencia_profissional = int(request.form["experiencia_profissional"])
        posse_casa = int(request.form["posse_casa"])
        quantia_emprestimo = float(request.form["quantia_emprestimo"])
        intencao = int(request.form["intencao_emprestimo"])
        taxa_juros = float(request.form["taxa_juros"]) / 100
        percentual_renda = float(request.form["percentual_renda"]) /100
        hist_credito = float(request.form["hist_credito"])
        pontuacao_credito = float(request.form["pontuacao_credito"])
        dividas_anteriores = int(request.form["dividas_anteriores"])

        caracteristicas = np.array(
            [
                [
                    idade,
                    genero,
                    educacao,
                    renda_anual,
                    experiencia_profissional,
                    posse_casa,
                    quantia_emprestimo,
                    intencao,
                    taxa_juros,
                    percentual_renda,
                    hist_credito,
                    pontuacao_credito,
                    dividas_anteriores,
                ]
            ]
        )
        predicao = model.predict(caracteristicas)
        mapeamento = {0: "Negado", 1: "Aceito"}
        resultado = mapeamento.get(predicao[0])
        return render_template("index.html", predicao=resultado)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
