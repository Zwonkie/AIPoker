import argparse
import os


def main():
    ap = argparse.ArgumentParser(description='live2 pilot -- headless capture->decision->action loop')
    ap.add_argument('--auto', action='store_true', help='execute clicks (default: recommend-only)')
    ap.add_argument('--window', help='window-title substring filter (default: auto-detect table)')
    ap.add_argument('--board-size', default='6-Max')
    ap.add_argument('--sims', type=int, default=2000)
    ap.add_argument('--probe', action='store_true', help='one capture+vision pass, save frame PNG, exit')
    ap.add_argument('--list', action='store_true', help='list candidate table windows, exit')
    args = ap.parse_args()

    if args.list:
        from live2.phpserver import capture
        for w in capture.list_windows(args.window):
            print(f"  hwnd={w['hwnd']}  {w['title']!r}")
        return

    from live2.pilot.loop import Pilot
    pilot = Pilot(auto=args.auto, window=args.window, board_size=args.board_size, sims=args.sims)
    if args.probe:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                           'diagnostics', 'pilot_probe.png')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pilot.probe(os.path.abspath(out))
        return
    pilot.run()


if __name__ == '__main__':
    main()
