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
import java.util.List;

import org.aiwolf.client.lib.ComingoutContentBuilder;
import org.aiwolf.client.lib.Content;
import org.aiwolf.client.lib.ContentBuilder;
import org.aiwolf.client.lib.DivinedResultContentBuilder;
import org.aiwolf.common.data.Agent;
import org.aiwolf.common.data.Judge;
import org.aiwolf.common.data.Role;
import org.aiwolf.common.data.Species;
import org.aiwolf.common.data.Talk;
import org.aiwolf.common.data.Vote;
import org.aiwolf.common.net.GameInfo;
import org.aiwolf.common.net.GameSetting;

/**
 * Class for the medium role
 * @author Victor and Ruben
 */
public class Medium extends BasePlayer {
	
	/** Boolean that indicat if the agent already divined */
	private boolean hasDivined;
	/** Last agent that was killed */
	private Judge lastExecution;
	/** Number of alive werewolf */
	private int aliveWerewolf;

	/**
	 * Getter of the name
	 */
	public String getName() {
		return "Medium";
	}

	/**
	 * Initialize function
	 */
	public void initialize(GameInfo _gameInfo, GameSetting _gameSetting) {
		this.init(_gameInfo);
		aliveWerewolf = 3;
		lastExecution = null;
	}

	/**
	 * Function called on every beginning of the day. 
	 * Decrease the trust coef of agent that voted against the last executed if not werewolf else increase it
	 */
	public void dayStart() {
		hasDivined = false;
		if(getCurrentGameInfo().getDay() < 2)
			return;
		lastExecution = getCurrentGameInfo().getMediumResult(); //Précision du type retiré (Judge lastExecution)
		Agent executed = lastExecution.getTarget();
		Species executedSpecies = lastExecution.getResult();
		List<Agent> executioners = new ArrayList<>();
		for (Vote vote : getCurrentGameInfo().getLatestVoteList()) {
			if (vote.getTarget().equals(executed))
				executioners.add(vote.getAgent());
		}
		if (executedSpecies.equals(Species.WEREWOLF))
			aliveWerewolf--;
		for (Agent agent : executioners) {
			if (executedSpecies.equals(Species.WEREWOLF))
				changeTrustCoef(agent, 10);
			else 
				changeTrustCoef(agent, -5);
		}
	}

	/**
	 * Talk like a villager and only come out if there is less than 10 villager or 3 werewolf.
	 * Will then say the species of the last executed agent after each turn.
	 */
	public String talk() {
		if(getCurrentGameInfo().getAliveAgentList().size() >= 10 && aliveWerewolf == 3){ //do not say anything until there are 10 player left and 2 or more wolf
			return villagerTalk();
		}

		if (!getIsCO()) {
			setIsCO(); //Make CO true
			ContentBuilder builder = new ComingoutContentBuilder(getMe(), Role.MEDIUM);
			return new Content(builder).getText();
		}
		// After CO, report species of last executed agent
		if(!hasDivined && getCurrentGameInfo().getExecutedAgent() != null){
			ContentBuilder builder = new DivinedResultContentBuilder(getCurrentGameInfo().getExecutedAgent(), lastExecution.getResult()) ; //say the species of the last executed
			hasDivined = true;
			return new Content(builder).getText();
		}

		return Content.OVER.getText();
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
					getComingoutMap().put(talker, content.getRole());
					if(talk.getText().contains("SEER"))
					changeTrustCoef(talker, 15);
					if(talk.getText().contains("MEDIUM"))
					changeTrustCoef(talker, -40);
					break;
				case DIVINED:
					if (talk.getText().contains("WEREWOLF")) {
						if (content.getTarget().equals(getMe()))
							changeTrustCoef(talker, -30);
						else
							changeTrustCoef(content.getTarget(), -10);
					} else {
						if (content.getTarget().equals(getMe()))
							changeTrustCoef(talker, 15);
						else
							changeTrustCoef(content.getTarget(), 10);
					}
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
				case IDENTIFIED: // Process identification report
					break;
				default:
					break;
			}
		}
		setTalkListHead(getCurrentGameInfo().getTalkList().size());
	}

	public Agent divine() {
		throw new UnsupportedOperationException();
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
}
