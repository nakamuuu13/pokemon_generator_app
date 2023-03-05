import pokemon_generater
from importer import *

openai.api_key = "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"
def generate_text(prompt, model, length):
    completions = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=length
    )
    message = completions.choices[0].text
    return message.strip()

app = Flask(__name__, static_folder='./templates/images')


# ルートディレクトリにアクセスがあった時の処理
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_generation_form', methods=["POST"])
def generation_form():
    
    # インスタンス化
    generator = pokemon_generater.Generator()
    
    generator.load_state_dict(torch.load('src/generator_mavg_training_model_gpu6.pth', map_location=torch.device('cpu')))
    # 推論モード
    generator.eval()

    # ノイズを入力
    z = torch.randn(1, 512*16)
    picture = generator.forward(z,8)
    picture = picture.detach().numpy()
    picture = np.clip(picture*255., 0, 255).astype(np.uint8)    
    #画像を出力
    for i in range(0,picture.shape[0]):
        output_fig = plt.figure()
        #dst[i]はこの時点で次元が[channel,height,width]となっているが、
        #画像として表示するにはtranspose(1,2,0)とすることで
        #[height,width,channel]に変換する必要がある
        image = (picture[i].transpose(1, 2, 0))

    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('src/templates/images/pokemon_generation.png', image)
    
    return render_template('index.html', image='images/pokemon_generation.png')

@app.route('/text_generation_form', methods=["POST"])
def text_generation_form():

    type_1 = request.form.getlist('type_1')
    type_2 = request.form.getlist('type_2')
    feature_1 = request.form.getlist('feature_1')
    feature_2 = request.form.getlist('feature_2')
    feature_3 = request.form.getlist('feature_3')

    prompt = "{}/{}タイプで、「{}」「{}」「{}」のキーワードだけで、フェイクポケモンの図鑑の説明を30字以内で生成してください。".format(type_1, type_2, feature_1, feature_2, feature_3)
    model = "text-davinci-003"
    length = 200

    text = generate_text(prompt, model, length)
    
    return render_template('index.html',image='images/pokemon_generation.png', text=text)


if __name__ == '__main__':
    # app.run(port=8888)
    app.run()