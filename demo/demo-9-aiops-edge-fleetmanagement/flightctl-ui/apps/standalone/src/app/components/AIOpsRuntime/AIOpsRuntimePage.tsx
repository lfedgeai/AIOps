import * as React from 'react';
import {
  Alert,
  Button,
  Flex,
  FlexItem,
  PageSection,
  Stack,
  StackItem,
  Title,
} from '@patternfly/react-core';
import ExternalLinkAltIcon from '@patternfly/react-icons/dist/js/icons/external-link-alt-icon';

import { useTranslation } from '@flightctl/ui-components/src/hooks/useTranslation';

/**
 * AIOps Runtime page — embeds the (entirely separate) agentic-AIOps runtime
 * harness UI as a page in Flight Control.
 *
 * The harness is an independent service (FastAPI + its own UI, own
 * ClickHouse, own model wiring) started with `aiops-runtime serve`; this
 * page only frames it. Keep the harness deployable/observable on its own —
 * do not move its logic into this app.
 *
 * The harness URL is set at build time via the AIOPS_RUNTIME_URL env var
 * (webpack DefinePlugin), defaulting to http://localhost:8085.
 */
const AIOpsRuntimePage = () => {
  const { t } = useTranslation();
  const harnessUrl = window.AIOPS_RUNTIME_URL || 'http://localhost:8085';

  return (
    <PageSection>
      <Stack hasGutter>
        <StackItem>
          <Flex justifyContent={{ default: 'justifyContentSpaceBetween' }}>
            <FlexItem>
              <Title headingLevel="h1">{t('AIOps Runtime')}</Title>
            </FlexItem>
            <FlexItem>
              <Button
                variant="link"
                component="a"
                href={harnessUrl}
                target="_blank"
                rel="noopener noreferrer"
                icon={<ExternalLinkAltIcon />}
                iconPosition="end"
              >
                {t('Open in a new tab')}
              </Button>
            </FlexItem>
          </Flex>
        </StackItem>
        <StackItem>
          <Alert
            variant="info"
            isInline
            title={t('Self-adopting closed-loop AIOps runtime (separate service)')}
          >
            {t(
              'The runtime harness runs as its own service (default {{ url }}): anomaly detection, RCA, and the four remediation modes (HIL / Governed / Direct / Image rebuild via bootc + FlightCtl rollout). It is embedded here for operator convenience only.',
              { url: harnessUrl },
            )}
          </Alert>
        </StackItem>
        <StackItem isFilled>
          <iframe
            src={harnessUrl}
            title={t('AIOps Runtime')}
            style={{
              width: '100%',
              height: 'calc(100vh - 260px)',
              minHeight: '480px',
              border: '1px solid var(--pf-t--global--border--color--default)',
              borderRadius: '4px',
              background: 'white',
            }}
          />
        </StackItem>
      </Stack>
    </PageSection>
  );
};

export default AIOpsRuntimePage;
