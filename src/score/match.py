NAME = {"near": "近端", "far": "远端"}
POINT = ("0", "15", "30", "40")


def _other(side):
    return "far" if side == "near" else "near"


def _serve_box(server, played):
    even = played % 2 == 0
    if server == "near":
        return "far_left" if even else "far_right"
    return "near_right" if even else "near_left"


class Match:
    def __init__(self, server="near", best_of=3):
        self.best_of = best_of
        self.need = best_of // 2 + 1
        self.server = server
        self.sets = {"near": 0, "far": 0}
        self.games = {"near": 0, "far": 0}
        self.pts = {"near": 0, "far": 0}
        self.tb = False
        self.done = False

    def _played(self):
        return self.pts["near"] + self.pts["far"]

    def _wins_game(self, side):
        a = self.pts[side] + 1
        b = self.pts[_other(side)]
        if self.tb:
            return a >= 7 and a - b >= 2
        return a >= 4 and a - b >= 2

    def _wins_set(self, side):
        if not self._wins_game(side):
            return False
        g = self.games[side] + 1
        h = self.games[_other(side)]
        if self.tb:
            return True
        return g >= 6 and g - h >= 2

    def _wins_match(self, side):
        return self._wins_set(side) and self.sets[side] + 1 >= self.need

    def tag(self):
        if self.done:
            return ""
        stakes = []
        for side in (self.server, _other(self.server)):
            if self._wins_match(side):
                return "赛点"
            if self._wins_set(side):
                stakes.append("盘点")
            elif self._wins_game(side) and side != self.server:
                stakes.append("破发点")
            elif self._wins_game(side):
                stakes.append("局点")
        if "盘点" in stakes:
            return "盘点"
        if "破发点" in stakes:
            return "破发点"
        if "局点" in stakes:
            return "局点"
        return ""

    def text(self):
        if self.tb:
            point = f"抢七 {self.pts['near']}-{self.pts['far']}"
        elif self.pts["near"] >= 3 and self.pts["far"] >= 3:
            if self.pts["near"] == self.pts["far"]:
                point = "平分"
            else:
                lead = "near" if self.pts["near"] > self.pts["far"] else "far"
                point = f"{NAME[lead]}占先"
        else:
            a = POINT[min(3, self.pts[self.server])]
            b = POINT[min(3, self.pts[_other(self.server)])]
            point = f"{a}-{b}"
        return (
            f"盘 {self.sets['near']}-{self.sets['far']}  "
            f"局 {self.games['near']}-{self.games['far']}  "
            f"{NAME[self.server]}发球  {point}"
        )

    def _finish_game(self, side):
        self.games[side] += 1
        self.pts = {"near": 0, "far": 0}
        g = self.games[side]
        h = self.games[_other(side)]
        if self.tb or (g >= 6 and g - h >= 2):
            was_tb = self.tb
            first = getattr(self, "tb_first", self.server)
            self.sets[side] += 1
            self.games = {"near": 0, "far": 0}
            self.tb = False
            if self.sets[side] >= self.need:
                self.done = True
                return
            self.server = _other(first if was_tb else self.server)
            return
        if self.games["near"] == 6 and self.games["far"] == 6:
            self.server = _other(self.server)
            self.tb = True
            self.tb_first = self.server
            return
        self.server = _other(self.server)

    def award(self, side):
        if self.done:
            return
        self.pts[side] += 1
        if self.tb:
            if self.pts[side] >= 7 and self.pts[side] - self.pts[_other(side)] >= 2:
                self._finish_game(side)
                return
            played = self._played()
            if played == 1 or (played > 1 and (played - 1) % 2 == 0):
                self.server = _other(self.server)
            return
        if self.pts[side] >= 4 and self.pts[side] - self.pts[_other(side)] >= 2:
            self._finish_game(side)


def infer_server(players=None, filled=None, bounces=None):
    votes = {"near": 0, "far": 0}
    n = min(20, len(filled or []), len(players or []))
    for i in range(n):
        box = (filled[i] or {}).get(1)
        if not box or len(box) < 4:
            continue
        u = (float(box[0]) + float(box[2])) * 0.5
        v = (float(box[1]) + float(box[3])) * 0.5
        best, best_d = None, 1e18
        for p in players[i]:
            b = p.get("bbox") or []
            side = p.get("side")
            if len(b) < 4 or side not in votes:
                continue
            if not (b[0] - 36 <= u <= b[2] + 36 and b[1] - 36 <= v <= b[3] + 36):
                continue
            d = (u - (b[0] + b[2]) * 0.5) ** 2 + (v - (b[1] + b[3]) * 0.5) ** 2
            if d < best_d:
                best, best_d = side, d
        if best:
            votes[best] += 1
    if votes["near"] != votes["far"] and max(votes.values()) >= 2:
        return "far" if votes["far"] > votes["near"] else "near"
    xs = {"near": [], "far": []}
    for fr in (players or [])[:30]:
        for p in fr:
            xyz = p.get("xyz")
            side = p.get("side")
            if xyz and side in xs:
                xs[side].append(abs(float(xyz[0])))
    med = {}
    for side, arr in xs.items():
        if len(arr) >= 5:
            arr = sorted(arr)
            med[side] = arr[len(arr) // 2]
    if len(med) == 2 and abs(med["near"] - med["far"]) >= 1.0:
        return "near" if med["near"] < med["far"] else "far"
    for b in bounces or []:
        z = b.get("serve_zone") or ""
        if z.startswith("near"):
            return "far"
        if z.startswith("far"):
            return "near"
    return "near"


def play_match(bounces, hits=None, server="near", best_of=3, confirm=45, n_frames=None, played=None):
    match = Match(server, best_of)
    events = [("bounce", b) for b in bounces] + [("hit", h) for h in (hits or [])]
    events.sort(key=lambda it: int(it[1]["frame_id"]))
    phase = "serve"
    faults = 0
    last_side = None
    points = []

    def live(i, fid, side):
        if played is None:
            if n_frames and fid + confirm >= int(n_frames):
                return True
            for kind, nxt in events[i + 1 :]:
                gap = int(nxt["frame_id"]) - fid
                if gap > confirm:
                    return False
                if kind == "hit" or nxt.get("in_singles"):
                    return True
            return False
        end = fid + confirm + 1
        for kind, nxt in events[i + 1 :]:
            nf = int(nxt["frame_id"])
            if nf <= fid:
                continue
            if kind == "bounce":
                end = min(end, nf)
                break
            if nf > fid + confirm:
                break
        if end <= fid + 1 or not side:
            return False
        hit_ok = False
        for kind, nxt in events[i + 1 :]:
            nf = int(nxt["frame_id"])
            if nf >= end:
                break
            if kind == "hit":
                hit_ok = True
                break
        if not hit_ok:
            return False
        return any(played(f, side) for f in range(fid + 1, end))

    for i, (kind, ev) in enumerate(events):
        if match.done:
            break
        fid = int(ev["frame_id"])
        if kind == "hit":
            if phase != "serve":
                last_side = None
            continue
        side = ev.get("side")
        inn = bool(ev.get("in_singles"))
        tag = match.tag()
        text = match.text()
        if phase == "serve":
            box = _serve_box(match.server, match._played())
            if inn and ev.get("serve_zone") == box:
                phase = "rally"
                faults = 0
                last_side = side
                continue
            if live(i, fid, side):
                if inn:
                    phase = "rally"
                    faults = 0
                    last_side = side
                continue
            if inn and side == match.server:
                continue
            faults += 1
            if faults < 2:
                continue
            winner = _other(match.server)
            reason = "双误"
            faults = 0
        elif not inn:
            if live(i, fid, side):
                continue
            winner = side
            reason = "出界"
        elif last_side and side == last_side:
            if live(i, fid, side):
                last_side = side
                continue
            winner = _other(side)
            reason = "二跳"
        else:
            last_side = side
            continue
        points.append(
            {
                "frame_id": fid,
                "winner": winner,
                "reason": reason,
                "tag": tag,
                "before": text,
            }
        )
        match.award(winner)
        points[-1]["after"] = match.text()
        phase = "serve"
        last_side = None
        faults = 0
    return {
        "server": server,
        "best_of": best_of,
        "points": points,
        "final": match.text(),
        "done": match.done,
    }
