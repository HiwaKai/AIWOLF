/**Copyright 2022 Moreno 

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and limitations under the License. **/

package org.testaiwolf.demo;

import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;


import org.aiwolf.client.lib.Content;
import org.aiwolf.client.lib.ContentBuilder;
import org.aiwolf.client.lib.DivinedResultContentBuilder;
import org.aiwolf.client.lib.EstimateContentBuilder;
import org.aiwolf.client.lib.XorContentBuilder;
import org.aiwolf.client.lib.ComingoutContentBuilder;
import org.aiwolf.common.data.Agent;
import org.aiwolf.common.data.Judge;
import org.aiwolf.common.data.Role;
import org.aiwolf.common.data.Species;
import org.aiwolf.common.data.Talk;
import org.aiwolf.common.net.GameInfo;
import org.aiwolf.common.net.GameSetting;

/**
 * Class for the seer role
 * @author Victor and Ruben
 */
public class Seer extends BasePlayer {

	/** Queue of divination */
	Deque<Judge> myDivinationQueue = new LinkedList<>();
	/** List of the liar agent */
	ArrayList<Agent> liar;
	/** Map of every agent and their corresponding species */
	Map<Agent, Species> divinedPlayer = new HashMap<>();

	/**
	 * Getter of the name
	 */
	public String getName() {
		return "Seer";
	}

	/**
	 * Initialize function
	 */
	public void initialize(GameInfo _gameInfo, GameSetting _gameSetting) {
		this.init(_gameInfo);
		myDivinationQueue.clear();
		divinedPlayer.clear();
		liar = new ArrayList<>();
	}

	/**
	 * Function called at the beginning of each day.
	 * Increase or deacrease the trust coef of the targeted agent depending on its species.
	 */
	public void dayStart() {
		Judge divination = getCurrentGameInfo().getDivineResult();
		if (divination != null) {
			myDivinationQueue.offer(divination);
			Agent target = divination.getTarget();
			Species result = divination.getResult();
			divinedPlayer.put(target, result);
			if (result == Species.HUMAN) {
				changeTrustCoef(target, 30);
			} else {
				setTrustCoef(target, 0);
			}
		}
	}

	/**
	 * Vote for the agent with the lowest trust coef
	 */
	public Agent vote() {
		List<Agent> candidates = new ArrayList<>();

		// Add the lowest trust coef agent (WereWolf)
		for (Agent agent : getAliveAgentListNotMe()) {
			if (getTrustCoef(agent) == 0) {
				candidates.add(agent);
			}
		}

		if (!candidates.isEmpty())
			return randomSelect(candidates);

		for (Agent agent : getAliveAgentListNotMe()) {
			if (getTrustCoef(agent) < 25 && !checkDivinedPlayer(agent)) {
				candidates.add(agent);
			}
		}

		if (!candidates.isEmpty())
			return randomSelect(candidates);

		for (Agent agent : getAliveAgentListNotMe()) {
			if (getTrustCoef(agent) < 50 && !checkDivinedPlayer(agent)) {
				candidates.add(agent);
			}
		}

		if (!candidates.isEmpty())
			return randomSelect(candidates);

		for (Agent agent : getAliveAgentListNotMe()) {
			if (getTrustCoef(agent) < 75 && !checkDivinedPlayer(agent)) {
				candidates.add(agent);
			}
		}

		if (candidates.isEmpty())
			return null;

		return randomSelect(candidates);
	}

	/**
	 * Talk function adapted between villager talk, added with special behaviour of the seer.
	 */
	public String talk() {
		ContentBuilder build;
		// Do CO if you find a werewolf in divination
		if (!getIsCO()) {
			if ((!myDivinationQueue.isEmpty() && myDivinationQueue.peekLast().getResult() == Species.WEREWOLF) || !liar.isEmpty()) {
				setIsCO(); //Make CO tru
				build = new ComingoutContentBuilder(getMe(), Role.SEER);
				return new Content(build).getText();
			} else {
				return villagerTalk();
			}
		}
		// After CO, report divination results that haven't yet been reported
		
		if (!myDivinationQueue.isEmpty()) {
			Judge divination = myDivinationQueue.pollLast();
			build = new DivinedResultContentBuilder(divination.getTarget(),
					divination.getResult());
			return new Content(build).getText();
		}
		// if no divination to say and false seer present
		if (liar.size()>0) {
			Agent actualLiar = liar.get(liar.size()-1);
			ContentBuilder werewolf =  new EstimateContentBuilder(actualLiar, Role.WEREWOLF);
			ContentBuilder possessed =  new EstimateContentBuilder(actualLiar, Role.POSSESSED);
			
			if(!getCurrentGameInfo().getAliveAgentList().contains(actualLiar)) //If the usurpator is already dead
				return Content.OVER.getText(); 

			if(getTrustCoef(actualLiar) == 0) //if TC is 0 then the agent is a werewolf
				build = werewolf;
			else if(divinedPlayer.containsKey(actualLiar)) //If TC is different than 0 and is known then it's a possessed
				build = possessed;
			else //If TC is different than 0 and is unknown then it's either werewolf or possessed
				build = new XorContentBuilder(new Content(werewolf), new Content(possessed));

			liar.remove(actualLiar);
			return new Content(build).getText();
		}
		return villagerTalk();
	}

	/**
	 * Update function called on every action
	 */
	public void update(GameInfo gameInfo) {
		setCurrentGameInfo(gameInfo);
		// Extract CO, divination reports, and identification reports
		// from GameInfo.talkList
		for (int i = getTalkListHead(); i < getCurrentGameInfo().getTalkList().size(); i++) {
			Talk talk = getCurrentGameInfo().getTalkList().get(i);
			Agent talker = talk.getAgent();
			if (talker == getMe()) {
				continue;
			}
			Content content = new Content(talk.getText()); // Parse utterances
			switch (content.getTopic()) {
				case COMINGOUT:
					// Process CO
					getComingoutMap().put(talker, content.getRole()); // decrease trust coef if comingout is seer
					if (talk.getText().contains("MEDIUM"))
						changeTrustCoef(talker, 15);
					if (talk.getText().contains("SEER")){
						changeTrustCoef(talker, -40);
						liar.add(talker);
					}
					break;
				case DIVINED: // Process divination report
					if (getAliveAgentListNotMe().contains(content.getTarget())) { //Ne pas réduire le trust coef si CO as Medium
						changeTrustCoef(talker, -25);
					}
					break;
				case IDENTIFIED: // Process identification report
					break;
				case VOTE:
					if (content.getTarget().equals(getMe())){
						changeTrustCoef(talker, -10);
						break;
					}
					if(getCurrentGameInfo().getDay() > 5){
						if(getTrustCoef(talker) <  getTrustCoef(content.getTarget()))
							changeTrustCoef(talker, -10);
						else
							changeTrustCoef(content.getTarget(), -10);
					}
					break;
				default:
					break;
			}
		}
		setTalkListHead(getCurrentGameInfo().getTalkList().size());
	}

	/**
	 * Divine function, select the agent with the lowest trust coef and which is not already divined
	 */
	public Agent divine() {
		List<Agent> candidates = new ArrayList<>();
		Agent ag;

		for (Agent agent : getAliveAgentListNotMe()) {
			if (getTrustCoef(agent) < 25 && !checkDivinedPlayer(agent)) {
				candidates.add(agent);
			}
		}

		if (!candidates.isEmpty()) {
			ag = randomSelect(candidates);
			divinedPlayer.put(ag, null);
			return ag;
		}

		for (Agent agent : getAliveAgentListNotMe()) {
			if (getTrustCoef(agent) < 50 && !checkDivinedPlayer(agent)) {
				candidates.add(agent);
			}
		}

		if (!candidates.isEmpty()) {
			ag = randomSelect(candidates);
			divinedPlayer.put(ag, null);
			return ag;
		}

		for (Agent agent : getAliveAgentListNotMe()) {
			if (getTrustCoef(agent) < 75 && !checkDivinedPlayer(agent)) {
				candidates.add(agent);
			}
		}

		if (candidates.isEmpty()) {
			return null;
		}
		
		ag = randomSelect(candidates);
		divinedPlayer.put(ag, null);
		return ag;
	}

	public Agent attack() {
		throw new UnsupportedOperationException();
	}

	public Agent guard() {
		throw new UnsupportedOperationException();
	}

	public String whisper() {
		throw new UnsupportedOperationException();
	}

	public void finish(){

	}

	/**
	 * Check wether an agent is in the divinedPlayer list
	 * 
	 * @param agent, agent that we want to check
	 * @return true if the agent is in the list, if not false
	 */
	private boolean checkDivinedPlayer(Agent agent) {
		return divinedPlayer.containsKey(agent);
	}
}
