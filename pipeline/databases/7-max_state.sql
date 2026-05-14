-- script that displays the max temperature of each state name ordered
SELECT `state`, MAX(`value`) AS max_temp
FROM temperatures
GROUP BY `state`
ORDER BY max_temp DESC;