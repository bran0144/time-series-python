import React from 'react'
import { useEffect } from 'react'
import { Box, Typography } from '@mui/material'
import { CircularProgress } from '@material-ui/core'
import Backdrop from '@material-ui/core/Backdrop'
import EditableElement from './EditableElement'
import apiConfig from './apiConfig'
import {
  rfc3339,
  formatGraphData,
  getServiceInfo,
  sortByElapseTime,
  calculateDate,
} from './helperFunction'
import axios from 'axios'

import InputLabel from '@material-ui/core/InputLabel'
import Select from '@material-ui/core/Select'

import MenuItem from '@material-ui/core/MenuItem'
import { LineGraph } from './LineGraph'

import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import TopSqlTable from './TopSqlTable'
import TopSqlIds from './TopSqlIds'
import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'

export default function PostgresMonthlyInsights(props) {
  const [loading, setLoading] = React.useState(true)

  const [monthOptions, setMonthOptions] = React.useState(['Last 30 days'])
  const [monthSelection, setMonthSelection] = React.useState('Last 30 days')
  const [dbServiceFail, setDbServiceFail] = React.useState({
    state: false,
    message: '',
  })
  const [databaseInfo, setDatabaseInfo] = React.useState({})

  const [postgresMonthlyInsightsData, setPostgresMonthlyInsightsData] = React.useState({})
  const [isPatroniLeader, setPatroniLeader] = React.useState({
    state: false
  })
  const [isPatroniStandby, setPatroniStandby] = React.useState({
    state: false
  })

  const [isUltron, setUltron] = React.useState({
    state: false
  })
 

// need to determine if DB is Ultron or not 
//      - just parse for the u? ie pgulx...
// need to determine if there is patroni and whether db is leader or standby
// need to determine version of db (if 13 and above or not )
//   - use Database Version endpoint under Postgres
  const [postgresMonthlyInsightsMetrics] = React.useState({
    CPU: {
      name: 'Host CPU',
      HostCPUCurrent30days: [],
      HostCPUCurrent7days: [],
      HostCPUStandby30days: [], 
      HostCPUStandby7days: [],
    },
    Memory: {
      name: 'Host Memory',
      HostMemoryCurrent30days: [],
      HostMemoryCurrent7days: [],
      HostMemoryStandby30days: [],
      HostMemoryStandby7days: [],
    },
    BufferHitRatio: {
      name: 'BufferHitRatio',
      BufferHitRatioCurrent30days: [],
      BufferHitRatioCurrent7days: [],
      BufferHitRatioStandby30days: [],
      BufferHitRatioStandby7days: [],
    },
    DBWaits: {
      name: 'DB Waits',
      DBWaitsCurrent30days: [],
      DBWaitsCurrent7days: [],
      DBWaitsStandby30days: [],
      DBWaitsStandby7days: [],
    },
    // these are calls to postgres, not influx and will need both host and dbname
    TopSqlPre13: {
      name: 'Top SQL for PG version older than 13',
      TopSQLLeaderPre13: [], 
      TopSQLStandbyPre13: [],
    },
    TopSq13andAbove: {
      name: 'Top SQL for PG version 13 and above',
      TopSQLLeader13Above: [],
      TopSQLStandby13Above: [],
      },
    // back to influx
    TopQueryPerformanceStats: {
      name: 'Top Query Performance Stats',
      TopQuery1Leader30days: [],
      TopQuery1Leader7days: [],
      TopQuery2Leader30days: [],
      TopQuery2Leader7days: [],
      TopQuery1Standby30days:[],
      TopQuery1Standby7days: [],
      TopQuery2Standby30days:[],
      TopQuery2Standby7days: [],
    },
    DbSessionInfo: {
      name: 'DB Session Info',
      DBSessionInfoLeader30days: [],
      DBSessionInfoLeader7days: [],
      DBSessionInfoStandby30days:[],
      DBSessionInfoStandby7days: [],
    },
    DiskIOPSUltron: {
      name: 'Disk IOPS Ultron',
      DiskIOPSUltronLeader30days: [],
      DiskIOPSUltronLeader7days: [],
      DiskIOPSUltronStandby30days:[],
      DiskIOPSUltronStandby7days: [],
    },
    DiskIOPSNonUltron: {
      name: 'Disk IOPS NonUltron',
      DiskIOPSNonUltronLeader30days: [],
      DiskIOPSNonUltronLeader7days:[],
      DiskIOPSNonUltronStandby30day: [],
      DiskIOPSNonUltronStandby7days: [],

    },
  })

  useEffect(() => {
    async function getPostgresMonthlyInsightsData() {
      setLoading(true)

      const current = new Date()
      current.setMonth(current.getMonth() - 1)

      const previousMonth = current.toLocaleString('default', {
        month: 'long',
        year: 'numeric',
      })

      if (!monthOptions.includes(previousMonth)) {
        setMonthOptions((oldArray) => [...oldArray, previousMonth])
      }

      let insightsData = {
        HostCPULeader30days: { data: [], error: ''},
        HostCPULeader7days: { data: [], error: ''},
        HostCPUStandby30days: { data: [], error: ''},
        HostCPUStandby7days: { data: [], error: ''},
        // HostMemoryLeader30days: { data: [], error: ''},
        // HostMemoryLeader7days: { data: [], error: ''},
        // HostMemoryStandby30days: { data: [], error: ''},
        // HostMemoryStandby7days: { data: [], error: ''},
        // BufferHitRatioLeader30days: { data: [], error: ''},
        // BufferHitRatioLeader7days: { data: [], error: ''},
        // BufferHitRatioStandby30days: { data: [], error: ''},
        // BufferHitRatioStandby7days: { data: [], error: ''},
        // DBWaitsLeader30days: { data: [], error: ''},
        // DBWaitsLeader7days: { data: [], error: ''},
        // DBWaitsStandby30days: { data: [], error: ''},
        // DBWaitsStandby7days: { data: [], error: ''},
        // TopSQLLeaderPre13: {data: [], error: ''},
        // TopSQLStandbyPre13: {data: [], error: ''},
        // TopSQLLeader13Above: {data: [], error: ''},
        // TopSQLStandby13Above: {data: [], error: ''},
        // TopQuery1Leader30days: {data: [], error: ''},
        // TopQuery1Leader7days: {data: [], error: ''},
        // TopQuery2Leader30days: {data: [], error: ''},
        // TopQuery2Leader7days: {data: [], error: ''},
        // TopQuery1Standby30days: {data: [], error: ''},
        // TopQuery1Standby7days: {data: [], error: ''},
        // TopQuery2Standby30days: {data: [], error: ''},
        // TopQuery2Standby7days: {data: [], error: ''},
        // DBSessionInfoLeader30days: {data: [], error: ''},
        // DBSessionInfoLeader7days: {data: [], error: ''},
        // DBSessionInfoStandby30days: {data: [], error: ''},
        // DBSessionInfoStandby7days: {data: [], error: ''},
        // DiskIOPSUltronLeader30days: {data: [], error: ''},
        // DiskIOPSUltronLeader7days: {data: [], error: ''},
        // DiskIOPSUltronStandby30days: {data: [], error: ''},
        // DiskIOPSUltronStandby7days: {data: [], error: ''},
        // DiskIOPSNonUltronLeader30days: {data: [], error: ''},
        // DiskIOPSNonUltronLeader7days: {data: [], error: ''},
        // DiskIOPSNonUltronStandby30days: {data: [], error: ''},
        // DiskIOPSNonUltronStandby7days: {data: [], error: ''},

      }

      const [monthStartTime, monthEndTime] = calculateDate(
        monthSelection,
        'month'
      )
      const [weekStartTime, weekEndTime] = calculateDate(monthSelection, 'week')

      let hostName = props.options.hostName

    //   dbname is 3rd column in dbnamemapping
      const DBInfo = axios
        .get(
          `${apiConfig.acme}/influxdb?dbtype=${
            props.options.dbType
          }&hostname=${hostName}&starttime=${rfc3339(
            monthStartTime
          )}&endtime=${rfc3339(monthEndTime)}&querytype=dbnamemapping`
        )
        .then((response) => {
          let serviceInfo = getServiceInfo(response, hostName)
          setDatabaseInfo(serviceInfo)
        })
        .catch((err) => {
          setDbServiceFail({ state: true, message: err.message })
        })

      const PatroniInfo = axios
      .get(
        `${apiConfig.acme}/influxdb?dbtype=${
          props.options.dbType
        }&hostname=${hostName}&starttime=${rfc3339(
          monthStartTime
        )}&endtime=${rfc3339(monthEndTime)}&querytype=patronileaderandstandby`
      )
      .then((response) => {
        let serviceInfo = getServiceInfo(response, hostName)
        setDatabaseInfo(serviceInfo)
      })
      .catch((err) => {
        setDbServiceFail({ state: true, message: err.message })
      }) 

      const PGVersion = axios
      .get(
        `${apiConfig.acme}/postgresdb?dbtype=${
          props.options.dbType
        }&host=${hostName}&dbname=${dbName}&starttime=${rfc3339(
          monthStartTime
        )}&endtime=${rfc3339(monthEndTime)}&querytype=dbversion`
      )
      .then((response) => {
        let serviceInfo = getServiceInfo(response, hostName)
        setDatabaseInfo(serviceInfo)
      })
      .catch((err) => {
        setDbServiceFail({ state: true, message: err.message })
      }) 

      const HostCPULeader30days = axios
        .get(
          `${apiConfig.acme}/influxdb?dbtype=${
            props.options.dbType
          }&hostname=${hostName}&starttime=${rfc3339(
            monthStartTime
          )}&endtime=${rfc3339(monthEndTime)}&querytype=hostcpu`
        )
        .then((response) => {
          insightsData.HostCPULeader30days.data = formatGraphData(response)
        })
        .catch((err) => {
          insightsData.HostCPULeader30days.error = err.message
        })

      const HostCPULeader7days = axios
        .get(
          `${apiConfig.acme}/influxdb?dbtype=${
            props.options.dbType
          }&hostname=${hostName}&starttime=${rfc3339(
            weekStartTime
          )}&endtime=${rfc3339(weekEndTime)}&querytype=hostcpu`
        )
        .then((response) => {
          insightsData.HostCPULeader7days.data = formatGraphData(response)
        })
        .catch((err) => {
          insightsData.HostCPULeader7days.error = err.message
        })

      // const HostCPUStandby30days = axios
      //   .get(
      //     `${apiConfig.acme}/influxdb?dbtype=${
      //       props.options.dbType
      //     }&hostname=${hostName}&starttime=${rfc3339(
      //       monthStartTime
      //     )}&endtime=${rfc3339(monthEndTime)}&querytype=hostcpu`
      //   )
      //   .then((response) => {
      //     insightsData.HostCPUStandby30days.data = formatGraphData(response)
      //   })
      //   .catch((err) => {
      //     insightsData.HostCPUStandby30days.error = err.message
      //   })

      // const HostCPUStandby7days = axios
      //   .get(
      //     `${apiConfig.acme}/influxdb?dbtype=${
      //       props.options.dbType
      //     }&hostname=${hostName}&starttime=${rfc3339(
      //       weekStartTime
      //     )}&endtime=${rfc3339(weekEndTime)}&querytype=hostcpu`
      //   )
      //   .then((response) => {
      //     insightsData.HostCPUStandby7days.data = formatGraphData(response)
      //   })
      //   .catch((err) => {
      //     insightsData.HostCPUStandby7days.error = err.message
      //   })

      

      axios
        .all([
            HostCPULeader30days,
            HostCPULeader7days,
            // HostCPUStandby30days,
            // HostCPUStandby7days,
            // HostMemoryLeader30days,
            // HostMemoryLeader7days,
            // HostMemoryStandby30days,
            // HostMemoryStandby7days,
            // BufferHitRatioLeader30days,
            // BufferHitRatioLeader7days,
            // BufferHitRatioStandby30days,
            // BufferHitRatioStandby7days,
            // DBWaitsLeader30days,
            // DBWaitsLeader7days,
            // DBWaitsStandby30days,
            // DBWaitsStandby7days,
            // TopSQLLeaderPre13,
            // TopSQLStandbyPre13,
            // TopSQLLeader13Above,
            // TopSQLStandby13Above,
            // TopQuery1Leader30days,
            // TopQuery1Leader7days,
            // TopQuery2Leader30days,
            // TopQuery2Leader7days,
            // TopQuery1Standby30days,
            // TopQuery1Standby7days,
            // TopQuery2Standby30days,
            // TopQuery2Standby7days,
            // DBSessionInfoLeader30days,
            // DBSessionInfoLeader7days,
            // DBSessionInfoStandby30days,
            // DBSessionInfoStandby7days,
            // DiskIOPSUltronLeader30days,
            // DiskIOPSUltronLeader7days,
            // DiskIOPSUltronStandby30days,
            // DiskIOPSUltronStandby7days,
            // DiskIOPSNonUltronLeader30days,
            // DiskIOPSNonUltronLeader7days,
            // DiskIOPSNonUltronStandby30days,
            // DiskIOPSNonUltronStandby7days,
        ])
        .then(() => {
          setPostgresMonthlyInsightsData(insightsData)
          setLoading(false)
        })
    }
    getPostgresMonthlyInsightsData()
  }, [
    monthOptions,
    monthSelection,
    setMonthSelection,
    props.options.hostName,
    props.options.dbType,
  ])

  const handleValueChange = (event) => {
    setMonthSelection(event.target.value)
  }

  return (
    <div key="postgres monthly insights" style={{ paddingRight: '24px' }}>
      <br></br>
      <br></br>
      <InputLabel key="Select Options">Select Options</InputLabel>
      <Select
        value={monthSelection}
        onChange={handleValueChange}
        label="Database Type"
      >
        {monthOptions.map((option) => (
          <MenuItem key={option} value={option}>
            {option}
          </MenuItem>
        ))}
      </Select>

      <Typography variant="h4" component="div" align="center" gutterBottom>
        {props.options.dbName[0].substring(0, 3)}
      </Typography>
      <Typography variant="h5" align="center" gutterBottom>
        Database Performance Insights Report
      </Typography>
      <Typography variant="h5" align="center" gutterBottom>
        {monthSelection}
      </Typography>

      {loading ? (
        <Backdrop style={{ zIndex: '1' }} open={loading}>
          <CircularProgress disableShrink />
        </Backdrop>
      ) : (
        <>
          {dbServiceFail.state ? (
            <Alert severity="warning" sx={{ mb: 2 }}>
              <AlertTitle>Host Names Error</AlertTitle>
              {dbServiceFail.message}
            </Alert>
          ) : (
            Object.keys(postgresMonthlyInsightsMetrics).map((item) => {
              return (
                <div key={item} style={{ paddingRight: '24px' }}>
                  {item === 'CPU' ? (
                    <>
                      {' '}
                      <Typography variant="h5" align="left" gutterBottom>
                        {item}
                      </Typography>
                      <Typography variant="h6" align="left">
                        {databaseInfo.node1.hostName} –{' '}
                        {databaseInfo.node1.instanceName} –{' '}
                        {databaseInfo.node1.serviceName}
                      </Typography>
                      {/* <Typography variant="h6" align="left" gutterBottom>
                    30 Day View
                  </Typography> */}
                      <div key="HostCPULeader30days">
                        <Card>
                          <CardContent>
                            {postgresMonthlyInsightsData.HostCPULeader30days.data.data !=
                            null ? (
                              <>
                                {postgresMonthlyInsightsData.HostCPULeader30days.error ===
                                '' ? (
                                  <LineGraph
                                    datasets={
                                        postgresMonthlyInsightsData.HostCPULeader30days.data
                                        .data[0]
                                    }
                                    labels={
                                        postgresMonthlyInsightsData.HostCPULeader30days.data
                                        .data[1]
                                    }
                                    title={item + ' 30 Day View'}
                                  />
                                ) : (
                                  <Typography component="div">
                                    <Box
                                      sx={{
                                        textAlign: 'center',
                                        fontWeight: 'bold',
                                        m: 1,
                                      }}
                                    >
                                      {'CPU Month'}
                                    </Box>
                                    <Box sx={{ textAlign: 'center', m: 1 }}>
                                      {postgresMonthlyInsightsData.HostCPULeader30days.error}
                                    </Box>
                                  </Typography>
                                )}
                              </>
                            ) : (
                              <>{'No Data Returned from API for CPU Month'}</>
                            )}
                          </CardContent>
                        </Card>
                        <br></br>
                      </div>
                      {/* <Typography variant="h6" align="left">
                    7 Day View
                  </Typography> */}
                      <div key="HostCPULeader30days">
                        <Card>
                          <CardContent>
                            {postgresMonthlyInsightsData.HostCPULeader7days.data.data !=
                            null ? (
                              <>
                                {postgresMonthlyInsightsData.HostCPULeader7days.error ===
                                '' ? (
                                  <LineGraph
                                    datasets={
                                        postgresMonthlyInsightsData.HostCPULeader7days.data
                                        .data[0]
                                    }
                                    labels={
                                        postgresMonthlyInsightsData.HostCPULeader7days.data
                                        .data[1]
                                    }
                                    title={item + ' 7 Day View'}
                                  />
                                ) : (
                                  <Typography component="div">
                                    <Box
                                      sx={{
                                        textAlign: 'center',
                                        fontWeight: 'bold',
                                        m: 1,
                                      }}
                                    >
                                      {'CPU Week'}
                                    </Box>
                                    <Box sx={{ textAlign: 'center', m: 1 }}>
                                      {postgresMonthlyInsightsData.HostCPULeader7days.error}
                                    </Box>
                                  </Typography>
                                )}
                              </>
                            ) : (
                              <>{'No Data Returned from API for CPU Week'}</>
                            )}
                          </CardContent>
                        </Card>
                        <br></br>
                      </div>
                      <Typography variant="h6" align="left" gutterBottom>
                        {databaseInfo.node2.hostName} –{' '}
                        {databaseInfo.node2.instanceName} –{' '}
                        {databaseInfo.node2.serviceName}
                      </Typography>
                      <Typography variant="h6" align="left" gutterBottom>
                        30 Day View
                      </Typography>
                      <div key="HostCPUStandby30days">
                        <Card>
                          <CardContent>
                            {postgresMonthlyInsightsData.HostCPUStandby30days.data.data !=
                            null ? (
                              <>
                                {postgresMonthlyInsightsData.HostCPUStandby30days.error ===
                                '' ? (
                                  <LineGraph
                                    datasets={
                                        postgresMonthlyInsightsData.HostCPUStandby30days.data
                                        .data[0]
                                    }
                                    labels={
                                        postgresMonthlyInsightsData.HostCPUStandby30days.data
                                        .data[1]
                                    }
                                    title={item + ' 30 Day View'}
                                  />
                                ) : (
                                  <Typography component="div">
                                    <Box
                                      sx={{
                                        textAlign: 'center',
                                        fontWeight: 'bold',
                                        m: 1,
                                      }}
                                    >
                                      {'CPU Month'}
                                    </Box>
                                    <Box sx={{ textAlign: 'center', m: 1 }}>
                                      {postgresMonthlyInsightsData.HostCPUStandby30days.error}
                                    </Box>
                                  </Typography>
                                )}
                              </>
                            ) : (
                              <>{'No Data Returned from API for CPU Month'}</>
                            )}
                          </CardContent>
                        </Card>
                        <br></br>
                      </div>
                      <Typography variant="h6" align="left" gutterBottom>
                        7 Day View
                      </Typography>
                      <div key="HostCPUStandby7days">
                        <Card>
                          <CardContent>
                            {postgresMonthlyInsightsData.HostCPUStandby7days.data.data !=
                            null ? (
                              <>
                                {postgresMonthlyInsightsData.HostCPUStandby7days.error ===
                                '' ? (
                                  <LineGraph
                                    datasets={
                                        postgresMonthlyInsightsData.HostCPUStandby7days.data
                                        .data[0]
                                    }
                                    labels={
                                        postgresMonthlyInsightsData.HostCPUStandby7days.data
                                        .data[1]
                                    }
                                    title={item + ' 7 Day View'}
                                  />
                                ) : (
                                  <Typography component="div">
                                    <Box
                                      sx={{
                                        textAlign: 'center',
                                        fontWeight: 'bold',
                                        m: 1,
                                      }}
                                    >
                                      {'CPU Week'}
                                    </Box>
                                    <Box sx={{ textAlign: 'center', m: 1 }}>
                                      {postgresMonthlyInsightsData.HostCPUStandby7days.error}
                                    </Box>
                                  </Typography>
                                )}
                              </>
                            ) : (
                              <>{'No Data Returned from API for CPU Week'}</>
                            )}
                          </CardContent>
                        </Card>
                        <br></br>
                      </div>
                      <Typography
                        variant="h5"
                        align="left"
                        editable="true"
                        gutterBottom
                      >
                        {item} Review and Recommendations:
                      </Typography>
                      <EditableElement>
                        <h1>Update Recommendations</h1>
                      </EditableElement>
                      <br></br>
                    </>
                  ) : (
                    <>
                      <Typography variant="h5" align="left" gutterBottom>
                        {postgresMonthlyInsightsMetrics[item].name}
                      </Typography>
                      {item === 'TopSql' ? (
                        <>
                          {' '}
                          <div key={item + 'Month'}>
                            <Card>
                              <CardContent>
                                {postgresMonthlyInsightsData[`${item}Month`].data
                                  .data != null ? (
                                  <>
                                    {postgresMonthlyInsightsData[`${item}Month`]
                                      .error === '' ? (
                                      <Card>
                                        <TopSqlTable
                                          data={
                                            postgresMonthlyInsightsData[`${item}Month`]
                                              .data.data[0]
                                          }
                                          tableOptions={
                                            postgresMonthlyInsightsData[`${item}Month`]
                                              .data.data[1]
                                          }
                                          title="Top SQl Last Month"
                                          comp="Monthly Insights"
                                        />
                                      </Card>
                                    ) : (
                                      <Typography component="div">
                                        <Box
                                          sx={{
                                            textAlign: 'center',
                                            fontWeight: 'bold',
                                            m: 1,
                                          }}
                                        >
                                          {postgresMonthlyInsightsMetrics[item].name +
                                            ' Month'}
                                        </Box>
                                        <Box sx={{ textAlign: 'center', m: 1 }}>
                                          {
                                            postgresMonthlyInsightsData[`${item}Month`]
                                              .error
                                          }
                                        </Box>
                                      </Typography>
                                    )}
                                  </>
                                ) : (
                                  <>
                                    {'No Data Returned from API for ' +
                                      postgresMonthlyInsightsMetrics[item].name +
                                      ' Month'}
                                  </>
                                )}
                              </CardContent>
                            </Card>
                            <br></br>
                          </div>
                          <div key={item + 'Week'}>
                            <Card>
                              <CardContent>
                                {postgresMonthlyInsightsData[`${item}Week`].data.data !=
                                null ? (
                                  <>
                                    {postgresMonthlyInsightsData[`${item}Week`]
                                      .error === '' ? (
                                      <TopSqlTable
                                        data={
                                          postgresMonthlyInsightsData[`${item}Week`]
                                            .data.data[0]
                                        }
                                        tableOptions={
                                          postgresMonthlyInsightsData[`${item}Week`]
                                            .data.data[1]
                                        }
                                        title="Top SQl Last Weak"
                                      />
                                    ) : (
                                      <Typography component="div">
                                        <Box
                                          sx={{
                                            textAlign: 'center',
                                            fontWeight: 'bold',
                                            m: 1,
                                          }}
                                        >
                                          {postgresMonthlyInsightsMetrics[item].name +
                                            ' Week'}
                                        </Box>
                                        <Box sx={{ textAlign: 'center', m: 1 }}>
                                          {
                                            postgresMonthlyInsightsData[`${item}Week`]
                                              .error
                                          }
                                        </Box>
                                      </Typography>
                                    )}
                                  </>
                                ) : (
                                  <>
                                    {'No Data Returned from API for ' +
                                      postgresMonthlyInsightsMetrics[item].name +
                                      ' Week'}
                                  </>
                                )}
                              </CardContent>
                            </Card>
                            <br></br>
                          </div>
                          <br></br>
                        </>
                      ) : (
                        <>
                          {item === 'TopSqlIds' ? (
                            <>
                              <Card>
                                <CardContent>
                                  {postgresMonthlyInsightsData[`${item}`].length > 0 ? (
                                    <TopSqlIds
                                      data={postgresMonthlyInsightsData[`${item}`]}
                                      monthSelection={monthSelection}
                                      options={props.options}
                                    />
                                  ) : (
                                    <>
                                      {'No Data Returned from API for ' +
                                        postgresMonthlyInsightsMetrics[item].name}
                                    </>
                                  )}
                                </CardContent>
                              </Card>
                              <br></br>
                            </>
                          ) : (
                            <>
                              <div key={item + 'Month'}>
                                <Card>
                                  <CardContent>
                                    {postgresMonthlyInsightsData[`${item}Month`].data
                                      .data != null ? (
                                      <>
                                        {postgresMonthlyInsightsData[`${item}Month`]
                                          .error === '' ? (
                                          <LineGraph
                                            datasets={
                                              postgresMonthlyInsightsData[
                                                `${item}Month`
                                              ].data.data[0]
                                            }
                                            labels={
                                              postgresMonthlyInsightsData[
                                                `${item}Month`
                                              ].data.data[1]
                                            }
                                            title={' 30 Day View'}
                                          />
                                        ) : (
                                          <Typography component="div">
                                            <Box
                                              sx={{
                                                textAlign: 'center',
                                                fontWeight: 'bold',
                                                m: 1,
                                              }}
                                            >
                                              {postgresMonthlyInsightsMetrics[item]
                                                .name + ' Month'}
                                            </Box>
                                            <Box
                                              sx={{ textAlign: 'center', m: 1 }}
                                            >
                                              {
                                                postgresMonthlyInsightsData[
                                                  `${item}Month`
                                                ].error
                                              }
                                            </Box>
                                          </Typography>
                                        )}
                                      </>
                                    ) : (
                                      <>
                                        {'No Data Returned from API for ' +
                                          postgresMonthlyInsightsMetrics[item].name +
                                          ' Month'}
                                      </>
                                    )}
                                  </CardContent>
                                </Card>
                                <br></br>
                              </div>
                              <div key={item + 'Week'}>
                                <Card>
                                  <CardContent>
                                    {postgresMonthlyInsightsData[`${item}Week`].data
                                      .data != null ? (
                                      <>
                                        {postgresMonthlyInsightsData[`${item}Week`]
                                          .error === '' ? (
                                          <LineGraph
                                            datasets={
                                              postgresMonthlyInsightsData[`${item}Week`]
                                                .data.data[0]
                                            }
                                            labels={
                                              postgresMonthlyInsightsData[`${item}Week`]
                                                .data.data[1]
                                            }
                                            title={' 7 Day View'}
                                          />
                                        ) : (
                                          <Typography component="div">
                                            <Box
                                              sx={{
                                                textAlign: 'center',
                                                fontWeight: 'bold',
                                                m: 1,
                                              }}
                                            >
                                              {postgresMonthlyInsightsMetrics[item]
                                                .name + ' Week'}
                                            </Box>
                                            <Box
                                              sx={{ textAlign: 'center', m: 1 }}
                                            >
                                              {
                                                postgresMonthlyInsightsData[
                                                  `${item}Week`
                                                ].error
                                              }
                                            </Box>
                                          </Typography>
                                        )}
                                      </>
                                    ) : (
                                      <>
                                        {'No Data Returned from API for ' +
                                          postgresMonthlyInsightsMetrics[item].name +
                                          ' Week'}
                                      </>
                                    )}
                                  </CardContent>
                                </Card>
                                <br></br>
                              </div>
                            </>
                          )}
                        </>
                      )}

                      <Typography
                        variant="h5"
                        align="left"
                        editable="true"
                        gutterBottom
                      >
                        {postgresMonthlyInsightsMetrics[item].name}: Review and
                        Recommendations:
                      </Typography>
                      <EditableElement>
                        <h1>Update Recommendations</h1>
                      </EditableElement>
                      <br></br>
                    </>
                  )}
                </div>
              )
            })
          )}
        </>
      )}
    </div>
  )
}
