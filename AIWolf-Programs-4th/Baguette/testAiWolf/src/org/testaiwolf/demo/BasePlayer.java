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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.aiwolf.client.lib.Content;
import org.aiwolf.client.lib.ContentBuilder;
import org.aiwolf.client.lib.VoteContentBuilder;
import org.aiwolf.common.data.Agent;
import org.aiwolf.common.data.Player;
import org.aiwolf.common.data.Role;
import org.aiwolf.common.net.GameInfo;
import org.aiwolf.common.net.GameSetting;

/**
 * The base class that is used by every role
 * @author Victor and Ruben
 */
public class BasePlayer implements Player {

    /** A random agent selected using a specific method, this agent will be used to vote */
    protected Agent badguy;

    /** The agent itself */
    private Agent me;
    
    /** The role of the agent */
    private Role myRole;
    
    /** The current state of the game */
    private GameInfo currentGameInfo;
    
    /** Boolean of the coming out of the agent */
    private boolean isCO;
    
    /**
     * Getter of the agent
     * @return the agent
     */
    protected Agent getMe(){
        return this.me;
    }

    /** Map of all the agent and their corresponding roles */
    private Map<Agent, Role> comingoutMap = new HashMap<>(); // Make sure initialized in initialize()
    private int talkListHead; // Make sure be set to 0 in dayStart()

    /** Map of all the agent and their corresponding trust coef */
    private Map<Agent, Integer> trustMap = new HashMap<>();
    
    /** Getter of the role of the agent */
    protected Role getMyRole(){
        return this.myRole;
    }

    /**
     * Initialisation function
     * @param _gameInfo
     */
    public void init(GameInfo _gameInfo){
        this.currentGameInfo = _gameInfo;
        this.me = _gameInfo.getAgent();
        this.myRole = _gameInfo.getRole();
        this.trustMap.clear();
        for(Agent agent : _gameInfo.getAgentList()){
            this.trustMap.put(agent, 50);
        }
        this.isCO = false;
    }

    /** 
     * Setter of the coming out boolean
     */
    protected void setIsCO(){
        this.isCO = true;
    }

    /**
     * Getter of the coming out boolean
     * @return isCo
     */
    protected boolean getIsCO(){
        return this.isCO;
    }

    /**
     * Boolean function used to find if an agent is dead or alive
     * @param _agent the agent 
     * @return true if the agent is found, false if not
     */
    protected boolean isAlive(Agent _agent){
        return currentGameInfo.getAliveAgentList().contains(_agent);
    }

    /**
     * Updater of the currentGameInfo
     * @param _currentGameInfo
     */
    protected void setCurrentGameInfo(GameInfo _currentGameInfo){
        this.currentGameInfo = _currentGameInfo;
    }

    /**
     * Getter of the currentGameInfo
     * @return
     */
    protected GameInfo getCurrentGameInfo(){
        return this.currentGameInfo;
    }
    
    /**
     * Select a random object in a generic list
     * @param <T>
     * @param list
     * @return the selected object
     */
    <T> T randomSelect(List<T> list) {
        if (list.isEmpty()) {
            return null;
        } else {
            return list.get((int) (Math.random() * list.size()));
        }
    }

    /**
     * Getter of the talkListHead
     * @return talkListHead
     */
    protected int getTalkListHead(){
        return this.talkListHead;
    }

    /**
     * Setter of the talkListHead
     * @param _talkListHead
     */
    protected void setTalkListHead(int _talkListHead){
        this.talkListHead = _talkListHead;
    }

    /**
     * Getter of the comingOutMap
     * @return
     */
    protected Map<Agent, Role> getComingoutMap(){
        return this.comingoutMap;
    }

    /**
     * Getter of the trustMap
     * @return
     */
    protected Map<Agent, Integer> getTrustMap(){
        return this.trustMap;
    }

    /**
     * Getter for the trustCoef
     * @param agent 
     * @return the trust coef of the agent wanted
     */
    protected int getTrustCoef(Agent agent){
        return this.trustMap.get(agent);
    }

    /**
     * Function used to change the trust coef of an agent to a specific value
     * @param agent  
     * @param coef 
     */
    protected void changeTrustCoef(Agent agent, int coef){
        if(getTrustCoef(agent)==0)
            return;
        if (getTrustCoef(agent)+coef>100){
            setTrustCoef(agent, 100);
        } else if (getTrustCoef(agent)+coef<1){
            setTrustCoef(agent, 1);
        } else {
            this.trustMap.put(agent, getTrustCoef(agent)+coef);
        }
    }

    /**
     * Setter of the trustCoef
     * @param agent
     * @param coef
     */
    protected void setTrustCoef(Agent agent, int coef){
        this.trustMap.put(agent, coef);
    }

    /**
     * Getter of the list of all alive agent without me
     * @return
     */
    protected List<Agent> getAliveAgentListNotMe(){
		List<Agent> list = new ArrayList<>();
		for(Agent agent : getCurrentGameInfo().getAliveAgentList()){ 
			if(!getMe().equals(agent))
				list.add(agent);
		}
		return list;
	}

    /**
     * Function to update the agent called Badguy
     */
    public void updateBadguy() {
        List<Agent> candidates = new ArrayList<>();

		for (Agent agent : getAliveAgentListNotMe()) {
			if (getTrustCoef(agent)<25) {
				candidates.add(agent);
			}
		}
		if (candidates.isEmpty()) {
			for (Agent agent : getAliveAgentListNotMe()) {
				if (getTrustCoef(agent)<50) {
					candidates.add(agent);
				}
			}
		}
		if (candidates.isEmpty()) {
			for (Agent agent : getAliveAgentListNotMe()) {
				if (getTrustCoef(agent)<75) {
					candidates.add(agent);
				}
			}
		}
		if (candidates.isEmpty()) {
			for (Agent agent : getAliveAgentListNotMe()) {
				if (getTrustCoef(agent)<=100) {
					candidates.add(agent);
				}
			}
		}
		this.badguy = randomSelect(candidates);
    }

    /**
     * Function used by all the villager to talk during the day
     * @return a prebuild string using the current badguy
     */
    public String villagerTalk(){
		updateBadguy();
        ContentBuilder builder = new VoteContentBuilder(this.badguy);
		return new Content(builder).getText();
    }

    @Override
    public String getName() {
        return null;
    }

    @Override
    public void update(GameInfo gameInfo) {

    }

    @Override
    public void initialize(GameInfo gameInfo, GameSetting gameSetting) {

    }

    @Override
    public void dayStart() {

    }

    @Override
    public String talk() {
        return null;
    }

    @Override
    public String whisper() {
        return null;
    }

    @Override
    public Agent vote() {
        return this.badguy;
    }

    @Override
    public Agent attack() {
        return null;
    }

    @Override
    public Agent divine() {
        return null;
    }

    @Override
    public Agent guard() {
        return null;
    }

    @Override
    public void finish() {

    }
}
