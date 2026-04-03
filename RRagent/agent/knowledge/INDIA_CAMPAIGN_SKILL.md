# INDIA CAMPAIGN SKILL — Japan Strategic Playbook

## Objective
Capture Calcutta (India) as the decisive victory condition.
India is worth 3 IPC and triggers Allied collapse if held.

---

## STAGE 1: Secure the Chinese Coastal Chain (Rounds 1-3)

### Mandatory Holdings
Keep all of the following Japanese-controlled at all times:
- **Shantung** [2 IPC] — northern coast anchor
- **Kiangsu** [3 IPC] — key coastal city, victory point
- **Kiangsi** [2 IPC] — gateway to Kwangtung
- **Kwangsi** [1 IPC] — southern flank protection

### Actions
1. **Do NOT overextend inland.** Skip Shensi, Hupeh, Szechwan — too far from the victory path.
2. Place 2-3 infantry in each coastal territory minimum.
3. Use transports to reinforce Kiangsi and coastal zones every round.
4. Advance: Anhwe → Kiangsi (overland), and use sea transport for Kwangtung.

### Transport Routing (Stage 1)
- Load at Japan → 6 Sea Zone
- Transit: 19 Sea Zone (adjacent to Kiangsu/Shantung)
- Land at Kiangsu or Shantung
- Immediately return empty transport to 6 Sea Zone same round

---

## STAGE 2: Capture Kwangtung & French Indo-China (Rounds 3-5)

### Why Kwangtung is Critical
Kwangtung (3 IPC) is a victory point AND the sea lane gateway to Southeast Asia.
Without Kwangtung, transports cannot efficiently reach French Indo-China.

### UK War Priority
**If Japan is at war with UK: capture Kwangtung FIRST before any other objective.**
- UK typically holds Kwangtung with 1-2 infantry in early game
- Attack with 4+ infantry + 1 artillery or 2 infantry + fighters = high win rate
- Use tool_predict_battle_odds to confirm ≥ 55% before attacking

### Advance Sequence
```
Kiangsi → Kwangtung (via land or amphibious from 36 Sea Zone)
          ↓
     French Indo-China
          ↓
        Burma
          ↓
         India
```

### Transport Routing (Stage 2)
- Load at Japan → 6 Sea Zone
- Transit: 36 Sea Zone (adjacent to French Indo-China and Kwangtung coast)
- Land troops at French Indo-China or Kwangtung coastal approach
- Return transport to 6 Sea Zone immediately after unloading

### Naval Positioning
- Move destroyers/cruisers to 37 Sea Zone (adjacent to India coast)
- This reduces UK IPC income and threatens India amphibious landing
- Battleships provide fire support for coastal assaults

---

## STAGE 3: India Push (Rounds 5+)

### Force Requirements Before Attacking India
- Ground forces in French Indo-China: minimum 5 units
- Air support: 2+ fighters within attack range
- Sea transport: 2+ transports staged at 37 or 41 Sea Zone
- Recommended attack stack: 4 infantry + 2 artillery + 1 tank + 2 fighters

### Attack Calculation
India typical defense: 2-3 UK infantry + 1 artillery
- Run tool_predict_battle_odds before committing
- Target: ≥ 65% win rate before launching India attack
- If win rate < 55%: delay, reinforce French Indo-China, try next round

### Transport Routing (Stage 3)
- Load tank + infantry at Japan → 6 Sea Zone
- Transit: 37 Sea Zone (adjacent to India / Burma coast)
- Land at Burma first → advance to India by land next round
- OR direct amphibious to India coast if sea zone is clear

### Air Support for India Attack
- Fighters stationed in Burma or French Indo-China can reach India (4 move)
- Tac bombers paired with fighters: +1 attack boost
- After India attack, fighters land at captured India
- Strategic bomber from Japan can bomb India factory to weaken UK production

---

## PURCHASE GUIDANCE (India Campaign)

### Round 1-2 (Setup)
Priority: 1 transport (if < 3) + 1 tank + 1 artillery + infantry to fill PUs
Goal: Build transport fleet to 3 ships minimum

### Round 3-5 (Build for South Push)  
Priority: 1 tank + 1 artillery + 2 infantry = fills 2 transports perfectly (12 PUs)
Skip fighters unless air coverage is thin (already have 2+ fighters)

### Round 5+ (India Strike)
Priority: tanks + artillery > infantry
Transport efficiency: 1 tank (2 slots) beats 2 infantry every time for India push

---

## COMMON MISTAKES TO AVOID

1. **Hoarding troops on Japan**: Japan is an island — units on Japan = zero combat power
2. **Not returning transports**: Empty transport must return to 6 SZ same NCM it unloads
3. **Skipping Kwangtung**: Cannot push south efficiently without controlling Kwangtung
4. **Attacking India too early**: Need 5+ units and 55%+ win rate; patience wins
5. **Wasting forces in central China**: Hupeh/Shensi/Szechwan are irrelevant to India path
6. **Not using aircraft**: Fighters can fly from Japan/Manchuria to support southern attacks

---

## SEA ZONE REFERENCE (India Route)

| Sea Zone | Adjacent Land Territories |
|----------|--------------------------|
| 6 SZ     | Japan (loading zone) |
| 19 SZ    | Kiangsu, Shantung |
| 20 SZ    | Manchuria, Korea |
| 36 SZ    | French Indo-China, Kwangtung coast |
| 37 SZ    | Burma, India, Malaya |
| 41 SZ    | India (Calcutta) approach |

---

## STRATEGIC SUMMARY

```
Round 1-2: Secure coast + build transports → Reinforce southern China
Round 3-4: Push to Kwangtung (mandatory if UK war) + take French Indo-China
Round 4-5: Stage 5+ units in FIC + position transports at 37 SZ
Round 5-6: Attack India with combined arms (ground + air + naval support)
```

**One transport cycling every round = 2 additional combat units delivered to the frontline.**
**Three transports cycling = decisive force multiplication. This is the path to victory.**
