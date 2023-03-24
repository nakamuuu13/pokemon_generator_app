import pokemon_generater
from importer import *

def generate_text(prompt, model, length):

    api_key = os.environ['OPEN_AI_API_KEY']
    if api_key is None:
        raise ValueError('API_KEY not set')

    openai.api_key = api_key
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
    
    generator.load_state_dict(torch.load('./generator_trained_model_gpu.pth', map_location=torch.device('cpu')))
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

    # リサイズ
    image = cv2.resize(image, (128, 128))
    # 画像書き込み用バッファ作成
    buf = io.BytesIO()
    image = Image.fromarray(image)
    image.save(buf, 'png')
    # バイナリデータを base64 でエンコードして utf-8 でデコード
    base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    # HTML 側の src の記述に合わせるために付帯情報付与する
    base64_data = 'data:image/png;base64,{}'.format(base64_str)

    return render_template('index.html', image=base64_data)

@app.route('/update')
def update_text():

    type_1 = request.form.getlist('type_1')
    type_2 = request.form.getlist('type_2')
    feature_1 = request.form.getlist('feature_1')
    feature_2 = request.form.getlist('feature_2')
    feature_3 = request.form.getlist('feature_3')

    prompt = "{}/{}タイプで, {}, {}, {}の特徴を持つ。このルールでフェイクポケモンの図鑑の説明を30字以内で生成してください。".format(type_1, type_2, feature_1, feature_2, feature_3)
    model = "text-davinci-003"
    length = 200

    new_text = generate_text(prompt, model, length)

    return new_text


if __name__ == '__main__':
    # app.run(port=8888)
    # app.run()
    app.run(host='0.0.0.0')