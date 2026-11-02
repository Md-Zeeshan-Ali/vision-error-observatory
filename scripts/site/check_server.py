import urllib.request
import urllib.error
import time
import subprocess
import sys

def test_server():
    proc = subprocess.Popen([sys.executable, "web_app/serve.py"])
    time.sleep(2)
    
    test_urls = [
        ("Root redirect", "http://localhost:8000/"),
        ("Main HTML", "http://localhost:8000/index.html"),
        ("Model Lab (3D Lab)", "http://localhost:8000/model_lab.html"),
        ("Favicon", "http://localhost:8000/favicon.ico"),
        ("Tailwind CSS", "http://localhost:8000/assets/css/tailwind.min.css"),
        ("Chart.js", "http://localhost:8000/assets/js/chart.min.js"),
        ("Three.js r128", "http://localhost:8000/assets/js/three.min.js"),
        ("OrbitControls.js", "http://localhost:8000/assets/js/OrbitControls.js"),
        ("FontAwesome CSS", "http://localhost:8000/assets/fontawesome/all.min.css"),
        ("VEO Redesign Data JS", "http://localhost:8000/data/app_data_v2.js"),
    ]
    
    all_passed = True
    print("\n--- RUNNING HTTP STATUS CHECKS ON REDESIGNED VEO PLATFORM ---")
    for name, url in test_urls:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                status = resp.status
                length = len(resp.read())
                print(f"[PASS] {name:<25} -> HTTP {status} (Bytes: {length:,})")
        except urllib.error.HTTPError as e:
            print(f"[FAIL] {name:<25} -> HTTP {e.code}")
            all_passed = False
        except Exception as e:
            print(f"[ERR]  {name:<25} -> {e}")
            all_passed = False

    # Terminate server
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except:
        proc.kill()
        
    print("\n--------------------------------------------------------------")
    if all_passed:
        print("[SUCCESS] All redesigned resources loaded with 0 errors and 0 404s!")
    else:
        print("[FAILED] Some requests failed.")
    return all_passed

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)
