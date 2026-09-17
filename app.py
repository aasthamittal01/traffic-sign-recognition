def simulator():

    # ========================================================
    # ORIGINAL TRAFFIC SIGN SIMULATION
    # Render the complete HTML/JavaScript simulation directly
    # instead of sending it through st.markdown().
    # ========================================================

    import base64
    from pathlib import Path
    import json

    # Real traffic-sign images from the GTSRB dataset
    sign_paths = {
        14: "/content/traffic_signs/Train/14/00014_00000_00000.png",   # STOP
        2:  "/content/traffic_signs/Train/2/00002_00000_00000.png",    # Speed 50
        20: "/content/traffic_signs/Train/20/00020_00000_00000.png",  # Curve right
        13: "/content/traffic_signs/Train/13/00013_00000_00000.png",  # Yield
        25: "/content/traffic_signs/Train/25/00025_00000_00000.png",  # Road work
        33: "/content/traffic_signs/Train/33/00033_00000_00000.png",  # Turn right
        34: "/content/traffic_signs/Train/34/00034_00000_00000.png"   # Turn left
    }

    sign_images = {}

    for class_id, path in sign_paths.items():

        try:

            encoded = base64.b64encode(
                Path(path).read_bytes()
            ).decode()

            sign_images[class_id] = (
                f"data:image/png;base64,{encoded}"
            )

        except:

            sign_images[class_id] = ""

    # Each sign has its own driving response
    scenarios = {

        14: {
            "name": "STOP",
            "action": "Vehicle slows down and stops",
            "target": 0,
            "behavior": "stop"
        },

        2: {
            "name": "Speed Limit 50",
            "action": "Vehicle reduces speed to 50 km/h",
            "target": 50,
            "behavior": "slow"
        },

        20: {
            "name": "Dangerous Curve Right",
            "action": "Vehicle slows and follows the right curve",
            "target": 35,
            "behavior": "curve-right"
        },

        13: {
            "name": "YIELD",
            "action": "Vehicle slows and continues carefully",
            "target": 25,
            "behavior": "yield"
        },

        25: {
            "name": "Road Work",
            "action": "Vehicle slows and proceeds carefully",
            "target": 30,
            "behavior": "slow"
        },

        33: {
            "name": "Turn Right Ahead",
            "action": "Vehicle slows and takes the right turn",
            "target": 30,
            "behavior": "turn-right"
        },

        34: {
            "name": "Turn Left Ahead",
            "action": "Vehicle slows and takes the left turn",
            "target": 30,
            "behavior": "turn-left"
        }
    }

    scenario_data = json.dumps(scenarios)
    image_data = json.dumps(sign_images)

    html = f"""
<!DOCTYPE html>

<html>

<head>

<style>

body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: #eef1f5;
    color: #111827;
}}

.dashboard {{
    width: 1100px;
    max-width: 100%;
    margin: 20px auto;
    background: white;
    border-radius: 18px;
    padding: 20px;
    box-sizing: border-box;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}}

.header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}}

.title {{
    font-size: 25px;
    font-weight: bold;
    color: #111827;
}}

.controls {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

select, button {{
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #999;
    font-size: 15px;
    color: #111827;
    background: white;
}}

button {{
    cursor: pointer;
    font-weight: bold;
    background: #f3f4f6;
}}

.speed-control {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
    color: #111827;
    font-weight: bold;
}}

.speed-control input {{
    width: 230px;
}}

#speedValue {{
    min-width: 70px;
}}

.scene {{
    position: relative;
    width: 100%;
    height: 430px;
    overflow: hidden;
    border-radius: 14px;
    background: linear-gradient(
        #9ed8ff 0%,
        #dff3ff 58%,
        #78ad62 58%
    );
}}

.road {{
    position: absolute;
    bottom: 0;
    width: 100%;
    height: 165px;
    background: #404348;
}}

.road-line {{
    position: absolute;
    top: 78px;
    width: 100%;
    border-top: 5px dashed #f5f5f5;
}}

.sign-pole {{
    position: absolute;
    right: 150px;
    bottom: 105px;
    width: 8px;
    height: 145px;
    background: #555;
}}

.sign {{
    position: absolute;
    right: 115px;
    bottom: 230px;
    width: 75px;
    height: 75px;
    background: white;
    border-radius: 10px;
    padding: 5px;
    object-fit: contain;
    filter: contrast(1.4) saturate(1.4);
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
}}

.car {{
    position: absolute;
    left: 40px;
    bottom: 52px;
    width: 120px;
    height: 55px;
    background: #26364a;
    border-radius: 18px 28px 10px 10px;
    box-shadow: 0 7px 10px rgba(0,0,0,0.25);
    transition: bottom 0.7s ease;
}}

.car::before {{
    content: "";
    position: absolute;
    width: 60px;
    height: 30px;
    left: 28px;
    top: -22px;
    background: #344e68;
    border-radius: 20px 20px 5px 5px;
}}

.wheel {{
    position: absolute;
    bottom: -10px;
    width: 25px;
    height: 25px;
    background: #151515;
    border-radius: 50%;
}}

.wheel.left {{
    left: 15px;
}}

.wheel.right {{
    right: 15px;
}}

.info {{
    display: flex;
    gap: 15px;
    margin-top: 15px;
}}

.card {{
    flex: 1;
    padding: 15px;
    border-radius: 12px;
    background: #f1f4f8;
}}

.label {{
    font-size: 13px;
    color: #374151;
}}

.value {{
    margin-top: 5px;
    font-size: 18px;
    font-weight: bold;
    color: #111827;
}}

.status {{
    margin-top: 15px;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
    font-size: 17px;
    font-weight: bold;
    color: #111827;
    background: #e5edff;
}}

</style>

</head>

<body>

<div class="dashboard">

    <div class="header">

        <div class="title">
            Traffic Sign Recognition & Driver Assistance
        </div>

        <div class="controls">

            <select id="scenario">

                <option value="14">STOP</option>

                <option value="2">
                    Speed Limit 50
                </option>

                <option value="20">
                    Dangerous Curve Right
                </option>

                <option value="13">
                    YIELD
                </option>

                <option value="25">
                    Road Work
                </option>

                <option value="33">
                    Turn Right Ahead
                </option>

                <option value="34">
                    Turn Left Ahead
                </option>

            </select>

            <button id="startButton">
                Start Simulation
            </button>

        </div>

    </div>


    <div class="speed-control">

        <span>Initial Speed:</span>

        <input
            type="range"
            id="initialSpeed"
            min="20"
            max="100"
            value="65"
            step="5"
        >

        <span id="speedValue">
            65 km/h
        </span>

    </div>


    <div class="scene">

        <div class="road">
            <div class="road-line"></div>
        </div>

        <div class="sign-pole"></div>

        <img
            id="sign"
            class="sign"
        >

        <div id="car" class="car">

            <div class="wheel left"></div>

            <div class="wheel right"></div>

        </div>

    </div>


    <div class="info">

        <div class="card">

            <div class="label">
                Detected Traffic Sign
            </div>

            <div id="signName" class="value">
                STOP
            </div>

        </div>


        <div class="card">

            <div class="label">
                Current Vehicle Speed
            </div>

            <div id="speed" class="value">
                65 km/h
            </div>

        </div>


        <div class="card">

            <div class="label">
                Driver Assistance
            </div>

            <div id="action" class="value">
                Vehicle slows down and stops
            </div>

        </div>

    </div>


    <div id="status" class="status">

        Select a scenario, choose the starting speed
        and start the simulation.

    </div>

</div>


<script>

const scenarios = {scenario_data};

const images = {image_data};


const scenarioSelect =
    document.getElementById("scenario");

const speedSlider =
    document.getElementById("initialSpeed");

const speedValue =
    document.getElementById("speedValue");

const sign =
    document.getElementById("sign");

const signName =
    document.getElementById("signName");

const speedDisplay =
    document.getElementById("speed");

const action =
    document.getElementById("action");

const status =
    document.getElementById("status");

const car =
    document.getElementById("car");

const startButton =
    document.getElementById("startButton");


let animationFrame = null;


speedSlider.addEventListener(
    "input",
    () => {{

        speedValue.innerText =
            speedSlider.value + " km/h";

    }}
);


function updateScenario() {{

    const id =
        scenarioSelect.value;

    const data =
        scenarios[id];

    sign.src =
        images[id];

    signName.innerText =
        data.name;

    action.innerText =
        data.action;

    speedDisplay.innerText =
        speedSlider.value + " km/h";

}}


scenarioSelect.addEventListener(
    "change",
    updateScenario
);


function startSimulation() {{

    cancelAnimationFrame(
        animationFrame
    );


    const id =
        scenarioSelect.value;

    const data =
        scenarios[id];


    const initialSpeed =
        Number(
            speedSlider.value
        );

    const targetSpeed =
        data.target;


    let currentSpeed =
        initialSpeed;

    let position = 40;

    let lastTime = null;

    let reachedSign = false;


    car.style.left =
        "40px";

    car.style.bottom =
        "52px";


    speedDisplay.innerText =
        initialSpeed + " km/h";

    status.innerText =
        "Vehicle approaching the traffic sign...";


    function animate(timestamp) {{

        if (!lastTime) {{

            lastTime =
                timestamp;

        }}


        const elapsed =
            (timestamp - lastTime) / 1000;

        lastTime =
            timestamp;


        if (
            position >= 540 &&
            !reachedSign
        ) {{

            reachedSign =
                true;

            status.innerText =
                data.name +
                " detected — " +
                data.action;

        }}


        if (reachedSign) {{

            if (
                currentSpeed >
                targetSpeed
            ) {{

                currentSpeed -=
                    25 * elapsed;

                if (
                    currentSpeed <
                    targetSpeed
                ) {{

                    currentSpeed =
                        targetSpeed;

                }}

            }}

            else if (
                currentSpeed <
                targetSpeed
            ) {{

                currentSpeed +=
                    20 * elapsed;

                if (
                    currentSpeed >
                    targetSpeed
                ) {{

                    currentSpeed =
                        targetSpeed;

                }}

            }}

        }}


        speedDisplay.innerText =
            Math.round(
                currentSpeed
            ) + " km/h";


        const movement =
            currentSpeed *
            elapsed *
            3.0;

        position +=
            movement;


        car.style.left =
            position + "px";


        if (
            data.behavior === "stop" &&
            position >= 700
        ) {{

            currentSpeed =
                0;

            speedDisplay.innerText =
                "0 km/h";

            car.style.left =
                "700px";

            status.innerText =
                "STOP detected — Vehicle has stopped.";

            return;

        }}


        if (
            data.behavior === "curve-right" &&
            position >= 570
        ) {{

            car.style.bottom =
                "95px";

            status.innerText =
                "Dangerous curve detected — " +
                "Vehicle slowing and following the right curve.";

        }}


        if (
            data.behavior === "turn-right" &&
            position >= 570
        ) {{

            car.style.bottom =
                "125px";

            status.innerText =
                "Right turn ahead — " +
                "Vehicle slowing and taking the right turn.";

        }}


        if (
            data.behavior === "turn-left" &&
            position >= 570
        ) {{

            car.style.bottom =
                "125px";

            status.innerText =
                "Left turn ahead — " +
                "Vehicle slowing and taking the left turn.";

        }}


        if (
            data.behavior !== "stop" &&
            position > 1150
        ) {{

            speedDisplay.innerText =
                Math.round(
                    currentSpeed
                ) + " km/h";

            status.innerText =
                data.name +
                " handled — Vehicle continues driving.";

            return;

        }}


        animationFrame =
            requestAnimationFrame(
                animate
            );

    }}


    animationFrame =
        requestAnimationFrame(
            animate
        );

}}


startButton.addEventListener(
    "click",
    startSimulation
);


updateScenario();

</script>

</body>

</html>
"""

    # THIS IS THE IMPORTANT FIX.
    # Do NOT use st.markdown(html, unsafe_allow_html=True)
    # for this simulation.

    components.html(
        html,
        height=720,
        scrolling=False
    )
