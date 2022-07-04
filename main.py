import base64
import pickle
from io import BytesIO

import torchvision
from jupyter_dash import JupyterDash
from dash import dcc 
from dash import html
from dash.dependencies import Input, Output

test_dset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=None)
def get_b64_img(idx):
    img, label = test_dset[idx]
    buff = BytesIO()
    img.save(buff, format='png')
    im_b64 = base64.b64encode(buff.getvalue()).decode('utf-8')
    return im_b64

if __name__ == '__main__':
    with open('./fig_v1.pkl', 'rb') as f:
        fig = pickle.load(f)

    app = JupyterDash(__name__)

    app.layout = html.Div([
        html.Div(id="output"),
        dcc.Graph(id="fig1", figure=fig)
    ])

    @app.callback(
        Output('output', 'children'),
        [Input('fig1', 'hoverData')])
    def display_image(hoverData):
        try:
            point = hoverData['points'][0]
            idx = point['customdata']
            im_b64 = get_b64_img(idx)
            value = 'data:image/png;base64,{}'.format(im_b64)
            return html.Img(src=value, height='100px')
        except:
            pass

    app.run_server(host='localhost', port=8005)

